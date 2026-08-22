# -*- coding: utf-8 -*-
"""SOL 30m 递送趋势策略的实盘适配模块。

将 SOL_30m_deliver 纯规则趋势跟踪（trend_strategy.py 核心逻辑）改造成
「计算最新一根 30m K 线的目标持仓 + 生成与 AI 决策同构的 TradingDecision」，
从而复用主流程的风控/执行/轮次记录链路，且全程不依赖任何 LLM。

与 AI 决策的分工差异：
- 分析师/决策者/确认者/复盘全部跳过；
- 方向信号 = 多尺度动量 sign × 波动率目标缩放（低杠杆，权益倍数表示持仓）；
- 目标仓位 ≈ 0 → 平仓；方向与现有持仓相反 → 先平后开；同方向 → 持有。

无未来函数：最优动量尺度 k 在「当前决策月起点」之前的历史上滚动选择
（与 trend_strategy.run 的回测口径一致）；最新一根 K 线的信号只依赖已收盘历史。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agents.schemas import OrderAction, OrderType, TradeInstruction, TradingDecision
from config import config
from trading.types import AccountInfo, Candle

logger = logging.getLogger(__name__)

# 与 SOL_30m_deliver/output/config.json 一致；文件缺失/异常时回退到这些默认值
DEFAULT_PARAMS = {
    "train_months": 24,
    "max_leverage": 5.0,
    "vol_target": 0.15,
    "vol_window": 20,
    "smooth_alpha": 0.50,
    "bars_per_day": 48,
}

# 动量尺度池与年化基准天数（与 trend_strategy.py 同口径）
K_POOL = [3, 5, 7, 10, 14, 20, 30, 40, 60, 90]
PPY = 365
FEE = 0.0004   # 单边手续费（pick_best_k 选择尺度时按成本口径，与回测一致）
SLIP = 0.0001

SOL_SYMBOL = "SOLUSDT"
INTERVAL = "30m"
MAX_KLINES_PER_REQ = 1500   # 币安 K 线单次上限

# 实盘参数
POS_THRESHOLD = 0.05    # |目标仓位| 低于此值视为无仓位（平仓）
STOP_VOL_MULT = 3.0     # 止损距离 = 3 × 日波动率（年化波动率 / sqrt(365)）
STOP_MIN_PCT = 0.02     # 止损距离下限
STOP_MAX_PCT = 0.12     # 止损距离上限
MARGIN_SAFETY = 0.98    # 开仓保证金占可用余额的上限比例（留出安全垫）


def _load_params() -> dict:
    """读取策略参数，优先使用 SOL_30m_deliver/output/config.json。"""
    path = (
        Path(__file__).resolve().parent.parent
        / "SOL_30m_deliver" / "output" / "config.json"
    )
    params = dict(DEFAULT_PARAMS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 %s 失败，使用默认策略参数: %s", path, exc)
    for k in ("train_months", "vol_window", "bars_per_day"):
        params[k] = int(params.get(k, DEFAULT_PARAMS[k]))
    for k in ("max_leverage", "vol_target", "smooth_alpha"):
        params[k] = float(params.get(k, DEFAULT_PARAMS[k]))
    return params


# ---------------------------------------------------------------------------
# 纯计算（与 trend_strategy.py 的 rolling_vol / pick_best_k / smooth 同口径）
# ---------------------------------------------------------------------------

def _rolling_vol(close: np.ndarray, window: int = 20) -> np.ndarray:
    """年化滚动波动率（shift(1) 避免用到当前收益的未来值）。"""
    r = np.concatenate([[0.0], np.diff(np.log(close))])
    s = pd.Series(r)
    vol = s.shift(1).rolling(window, min_periods=1).std() * np.sqrt(PPY)
    return np.nan_to_num(vol.values, nan=np.inf)


def _pick_best_k(close: np.ndarray, decision_idx: int, pool: list[int]) -> int:
    """在截止 decision_idx 的历史上选最优动量尺度 k（扣除成本后夏普最大）。"""
    h = close[: decision_idx + 1]
    best_k, best_sharpe = pool[0], -1e9
    for k in pool:
        if len(h) <= k + 1:
            continue
        ret = np.log(h[k:] / h[:-k])
        pos = np.sign(ret)
        turn = np.abs(np.diff(np.concatenate([[0], pos])))
        r = np.diff(np.log(h))
        net = pos * r[k - 1:] - turn * (FEE + SLIP)
        if net.size <= 1 or np.std(net) == 0:
            continue
        sharpe = np.mean(net) / np.std(net) * np.sqrt(PPY)
        if sharpe > best_sharpe:
            best_sharpe, best_k = sharpe, k
    return best_k


def _smooth(pos: np.ndarray, alpha: float = 0.25, max_step: Optional[float] = None) -> np.ndarray:
    """EMA 平滑（与 trend_strategy.smooth 一致）。"""
    out = np.empty_like(pos)
    out[0] = pos[0]
    for i in range(1, len(pos)):
        t = out[i - 1] * (1 - alpha) + pos[i] * alpha
        if max_step is not None:
            t = np.clip(t, out[i - 1] - max_step, out[i - 1] + max_step)
        out[i] = t
    return out


def compute_target(close: np.ndarray, params: dict) -> dict:
    """复现 trend_strategy.run 的位置计算，返回最新一根的目标信息。

    返回 dict：
      target_pos  最新目标仓位（权益倍数，正=多 / 负=空）
      vol_annual  最新年化波动率
      best_k      当前决策月使用的最优动量尺度（根数）
      sign        当前动量方向（1 / -1 / 0）
    """
    bpd = int(params["bars_per_day"])
    train_days = int(params["train_months"] * 30 * bpd)
    month_bars = int(30 * bpd)
    k_pool = [max(2, int(round(k * bpd))) for k in K_POOL]
    vol = _rolling_vol(close, int(params["vol_window"] * bpd))

    pos = np.zeros(len(close))
    k = k_pool[0]
    sign = 0.0
    i = train_days
    while i < len(close):
        k = _pick_best_k(close, i, pool=k_pool)
        j = i
        while j < len(close) and (j - i) < month_bars:
            ref = close[max(0, j - k)]
            sign = 1.0 if close[j] > ref else (-1.0 if close[j] < ref else 0.0)
            v = vol[j]
            target = params["vol_target"] / v if (v > 0 and np.isfinite(v)) else 0.0
            pos[j] = np.clip(sign * target, -params["max_leverage"], params["max_leverage"])
            j += 1
        i = j
    pos = _smooth(pos, params["smooth_alpha"])
    # 与更新后的回测同口径：EMA 平滑后再就近取整到整数杠杆档 {0,±1,...,±max_leverage}，
    # 满足真实交易杠杆必须为整数的约束，且从不超过 max_leverage。
    pos = np.clip(np.round(pos), -params["max_leverage"], params["max_leverage"])

    v_now = vol[-1]
    return {
        "target_pos": float(pos[-1]),
        "vol_annual": float(v_now) if np.isfinite(v_now) else 0.0,
        "best_k": k,
        "sign": sign,
    }


# ---------------------------------------------------------------------------
# 行情获取（向前分页回溯 30m K 线）
# ---------------------------------------------------------------------------

def fetch_30m_klines(client, symbol: str = SOL_SYMBOL, bars: Optional[int] = None) -> list[Candle]:
    """向前分页拉取 30m K 线（币安单次上限 1500，用 endTime 回溯），按时间升序。

    bars 默认取「训练期 + 当前决策月 + 动量/波动率窗口缓冲」根，保证与回测口径一致。
    """
    params = _load_params()
    bpd = int(params["bars_per_day"])
    if bars is None:
        # 训练期(24月) + 当前决策月(1月) + 最大动量尺度/波动率窗口缓冲
        bars = (
            int(params["train_months"] * 30 * bpd)
            + int(30 * bpd)
            + int(90 * bpd)
        )
    out: list[Candle] = []
    end_time: Optional[int] = None
    while len(out) < bars:
        page = client.get_klines(symbol, INTERVAL, MAX_KLINES_PER_REQ, end_time=end_time)
        if not page:
            break
        if out:
            # 防重叠：endTime 分页边界可能返回与已取部分重叠的K线，丢弃旧的重复段
            page = [c for c in page if c.open_time < out[0].open_time]
            if not page:
                break
        out = page + out
        end_time = out[0].open_time - 1
    return out[-bars:]


# ---------------------------------------------------------------------------
# 策略主体：目标持仓 → TradingDecision
# ---------------------------------------------------------------------------

class Sol30mStrategy:
    """SOL 30m 递送趋势策略。仅操作 SOLUSDT，不调用 LLM。"""

    def __init__(self, client) -> None:
        self.client = client
        self.params = _load_params()
        self.symbol = SOL_SYMBOL

    def compute_signal(self) -> dict:
        candles = fetch_30m_klines(self.client, self.symbol)
        if len(candles) < 100:
            raise RuntimeError(f"SOL 30m K线数据不足（{len(candles)} 根）")
        # 关键口径：丢弃「尚未收盘」的最新一根 K 线（其 close 是实时变动的未完成价，
        # 且它本身就是本轮刚开盘的新 K 线），保证信号只基于「已收盘」的 30m K 线。
        # 这样每根 K 线收盘后计算一次，close[-1] 落在刚收盘的那根上，与回测口径一致。
        now_ms = int(time.time() * 1000)
        closed = [c for c in candles if c.close_time <= now_ms]
        if not closed:
            raise RuntimeError("SOL 30m 无已收盘 K 线")
        close = np.asarray([c.close for c in closed], dtype=float)
        info = compute_target(close, self.params)
        info["price"] = float(close[-1])
        info["bars"] = len(closed)
        info["last_open_time"] = closed[-1].open_time
        return info

    def decide(
        self,
        account: AccountInfo,
        price_map: dict[str, float],
        symbol_info_map: dict,
    ) -> TradingDecision:
        """计算信号并翻译为 TradingDecision（空指令=维持原状）。"""
        sig = self.compute_signal()
        price = price_map.get(self.symbol) or sig["price"]
        equity = account.margin_balance if account.margin_balance > 0 else account.total_balance
        target = sig["target_pos"]
        bars = sig["bars"]
        vol_a = sig["vol_annual"]
        # 交易所杠杆：按回测配置 max_leverage 作为上限/保证金换算倍率（交易所只接受整数）。
        # 但「有效杠杆」= 名义价值/权益 = |目标仓位|，随波动率变化（回测均值约 0.8x），
        # 并非恒为 max_leverage；max_leverage 只是持仓规模上限。
        lev = max(1, int(self.params["max_leverage"]))
        eff_lev = abs(target)

        # 当前 SOL 持仓
        pos = next((p for p in account.positions if p.symbol == self.symbol), None)
        cur_qty = abs(pos.position_amt) if pos else 0.0
        cur_side = (
            "LONG" if pos and pos.position_amt > 0
            else ("SHORT" if pos and pos.position_amt < 0 else None)
        )

        # 止损距离：由年化波动率折算日波动率
        if vol_a > 0 and np.isfinite(vol_a):
            daily_vol = vol_a / np.sqrt(PPY)
            stop_pct = float(np.clip(STOP_VOL_MULT * daily_vol, STOP_MIN_PCT, STOP_MAX_PCT))
        else:
            stop_pct = 0.04

        instructions: list[TradeInstruction] = []
        notes: list[str] = []

        def _open(action: OrderAction, margin: float) -> TradeInstruction:
            side = "多" if action == OrderAction.OPEN_LONG else "空"
            sl = price * (1 - stop_pct) if action == OrderAction.OPEN_LONG else price * (1 + stop_pct)
            return TradeInstruction(
                symbol=self.symbol,
                action=action,
                order_type=OrderType.MARKET,
                margin=round(max(margin, 0.0), 4),
                leverage=lev,
                stop_loss=round(sl, 4),
                reason=(
                    f"SOL_30m 趋势策略：目标仓位 {target:+.3f}（{side}头，权益倍数）"
                    f"，动量尺度 k={sig['best_k']}，年化波动率 {vol_a:.2f}"
                    f"，止损 {stop_pct * 100:.1f}%（{sl:.4f}），有效杠杆 {eff_lev:.2f}x"
                ),
            )

        # ---- 翻译为指令 ----
        if abs(target) < POS_THRESHOLD:
            # 信号消失 → 平仓（若有持仓）
            if cur_qty > 0:
                instructions.append(TradeInstruction(
                    symbol=self.symbol, action=OrderAction.FLATTEN, order_type=None,
                    reason=(
                        f"SOL_30m 趋势策略：目标仓位 {target:+.3f} ≈ 0（动量信号消失），"
                        f"平掉当前 {cur_side} 持仓 {cur_qty}"
                    ),
                ))
                notes.append(f"信号转平 → 平仓 {cur_qty} {cur_side}")
            else:
                notes.append("信号转平，无持仓，HOLD")
        else:
            target_side = "LONG" if target > 0 else "SHORT"
            # 开仓保证金 = 目标名义（|target|×权益）/ 杠杆，并受可用余额安全上限约束
            target_notional = abs(target) * equity
            max_margin = max(0.0, account.available_balance) * MARGIN_SAFETY
            margin = min(target_notional / lev, max_margin)
            # 保证金下限 = 当前杠杆下的最小保证金（SOL 最小名义按确认放宽为 5.83/1x），
            # 保证下单名义至少满足交易所最小可成交要求，避免被风控按最小名义拦截。
            min_margin_lev = config.sol_min_notional / lev
            margin = max(margin, min_margin_lev)
            if cur_qty <= 0:
                action = OrderAction.OPEN_LONG if target > 0 else OrderAction.OPEN_SHORT
                instructions.append(_open(action, margin))
                notes.append(f"信号 {target_side} → 开仓 {margin:.2f}U（有效杠杆 {eff_lev:.2f}x，目标名义 {target_notional:.2f}U）")
            elif cur_side != target_side:
                # 方向翻转 → 先平后开（风控拆分保证平仓先执行）
                instructions.append(TradeInstruction(
                    symbol=self.symbol, action=OrderAction.FLATTEN, order_type=None,
                    reason=(
                        f"SOL_30m 趋势策略：方向翻转 {cur_side} → {target_side}，"
                        f"先平掉旧仓再按新方向开仓"
                    ),
                ))
                action = OrderAction.OPEN_LONG if target > 0 else OrderAction.OPEN_SHORT
                instructions.append(_open(action, margin))
                notes.append(f"方向翻转 {cur_side}→{target_side} → 平旧开新（{margin:.2f}U，有效杠杆 {eff_lev:.2f}x）")
            else:
                notes.append(f"信号 {target_side}，与现有持仓一致，HOLD")

        # 上下文摘要
        assessment = (
            f"SOL 30m 递送趋势策略：目标仓位 {target:+.3f}（权益倍数），"
            f"动量尺度 k={sig['best_k']}，年化波动率 {vol_a:.2f}，止损距离 {stop_pct * 100:.1f}%。"
            f" {'；'.join(notes) or '无操作'}"
        )
        risk_notes = (
            f"规则策略：SOL 有效杠杆=目标仓位（权益倍数，随波动率变化，非恒为 {lev}x；"
            f"交易所杠杆按 max_leverage={lev}x 设置，保证金=目标名义/杠杆，且不超过可用余额"
            f"的 {MARGIN_SAFETY * 100:.0f}%），止损按 {STOP_VOL_MULT}×日波动率折算"
            f"（{stop_pct * 100:.1f}%）自动设置；不调用 LLM、不设条件唤醒；"
            f"同方向持有、信号翻转先平后开。数据根数 {bars}。"
        )
        return TradingDecision(
            market_assessment=assessment,
            instructions=instructions,
            risk_notes=risk_notes,
            watch_conditions=[],
        )
