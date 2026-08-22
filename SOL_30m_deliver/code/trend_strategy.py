#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纯规则趋势跟踪策略（不依赖神经网络）
====================================
核心思想：用多尺度动量（价格相对 N 日前的变化方向）作为趋势信号，
叠加波动率目标缩放 + 低杠杆 + 持仓平滑，让净值成为"接近平缓的上升曲线"。

每月滚动重训（无未来函数）：
  - 每个决策月，只用"截止当月末之前"的历史拟合"最优动量尺度 k"（在训练段上
    以扣除成本后的夏普最大为准则），然后把该尺度应用到下月，产出样本外方向。
  - 样本外段的持仓/收益只依赖当月已定的 (k, 波动率, 持仓平滑)，无前瞻。

输出：净值 / 回撤 / 持仓 PNG 与指标打印。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


K_POOL = [3, 5, 7, 10, 14, 20, 30, 40, 60, 90]
FEE = 0.0004   # 单边手续费 0.04%
SLIP = 0.0001  # 滑点 0.01%
PPY = 365


def load_ts(csv):
    df = pd.read_csv(csv, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    close = df['close'].astype(float).values
    return df.index, close


def run_flat(csv, train_months=24, target_annual=0.40, vol_window=20,
             max_step=None, out_png=None, bars_per_day=1):
    """
    常数漂移目标法：让净值"趋近一条恒定斜率的上升直线"。

    原理：趋势方向(多尺度动量)给出 sign；再用"过去1个季度的单位暴露年化收益"
    按月更新一个缓慢变化的杠杆倍率 g，使外推年化收益≈target_annual。
    倍率 g 有下限(避免弱趋势时仓位归零导致净值横盘)与上限(控风险)，且逐月
    指数平滑，保证净值逐步缓升、无台阶拉升。

    无未来函数：每月唯一的当前信息是"截止当月的历史收益"，不触碰未来数据。
    bars_per_day: 每根K线对应的自然日数(日线=1, 30m=48)，用于统一时间口径。
    """
    idx, close = load_ts(csv)
    n = len(close)
    bpd = bars_per_day
    ppy = int(365 * bpd)
    train_days = int(train_months * 30 * bpd)
    vol = rolling_vol(close, int(vol_window * bpd))
    r = np.concatenate([[0.0], np.diff(np.log(close))])
    k_pool = [max(2, int(round(k * bpd))) for k in K_POOL]

    # 单位风险暴露 = sign / vol（波动率目标：每单位仓位带来恒定风险）
    unit = np.zeros(n)
    i = train_days
    month_bars = int(30 * bpd)
    while i < n:
        k = pick_best_k(close, i, pool=k_pool)
        j = i
        while j < n and (j - i) < month_bars:
            ref = close[max(0, j - k)]
            sign = 1.0 if close[j] > ref else (-1.0 if close[j] < ref else 0.0)
            v = vol[j]
            unit[j] = sign * (1.0 if (not np.isfinite(v) or v <= 0) else 1.0 / v)
            j += 1
        i = j
    unit = np.where(np.isnan(unit) | np.isinf(unit), 0.0, unit)

    pos = np.zeros(n)
    g = 1.0                      # 当前杠杆倍率（慢变量）
    ret_lookback_days = 90       # 一个季度的收益来估计"单位暴露年化"
    month = month_bars
    prev_reb = train_days
    now_reb = train_days + month
    while now_reb < n:
        # 截止 now_reb-1，最近一个季度单位暴露的日收益，年化后作为"单位暴露年化收益"
        lo = max(train_days, now_reb - int(ret_lookback_days * bpd))
        seg_unit = unit[lo:now_reb]
        seg_r = r[lo:now_reb]
        daily = float(np.sum(seg_unit * seg_r)) / max(len(seg_unit), 1)
        annual_unit = daily * ppy
        # 目标倍率：历史单位收益为正→放大到年化目标；否则用下限保底(避免横盘)
        g_target = (target_annual / annual_unit) if annual_unit > 1e-4 else 0.0
        g_target = float(np.clip(g_target, 0.10, 4.0))
        g = 0.8 * g + 0.2 * g_target               # 逐月平滑，避免杠杆跳变
        # 应用到下一个月
        pos[now_reb - month:now_reb] = g * unit[now_reb - month:now_reb]
        prev_reb = now_reb
        now_reb += month
    # 尾段贴现
    if prev_reb < n:
        pos[prev_reb:] = g * unit[prev_reb:]

    # 最终平滑：EMA + 可选最大日仓变
    pos = smooth(pos, 0.35, max_step=max_step)
    pos[:train_days] = 0.0

    _, net, eq, cost = backtest_positions(pos, close)
    m = metrics(eq, ppy=ppy)
    m['mean_leverage'] = float(np.mean(np.abs(pos)))
    m['start'] = str(idx[train_days].date())
    m['eq'] = eq
    m['idx'] = idx
    m['pos'] = pos
    m['close'] = close
    m['net'] = net
    m['cost'] = cost

    if out_png:
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1, 1]})
        ax_eq, ax_dd, ax_pos = axes
        ax_eq.plot(idx, np.exp(eq), color='#1FA28F', lw=1.6)
        ax_eq.set_title(f"{os.path.basename(csv)} · 常数漂移目标法(对数) 年化={m['annual_return']*100:.0f}% "
                        f"回撤={m['max_drawdown']*100:.0f}% 夏普={m['sharpe']:.2f} linR2={m['lin_r2']:.2f}")
        ax_eq.set_yscale('log'); ax_eq.grid(alpha=0.3)
        dd = -(eq - np.maximum.accumulate(eq))
        ax_dd.fill_between(idx, 0, dd * 100, color='#E8463A', alpha=0.5)
        ax_dd.set_ylabel('回撤%'); ax_dd.grid(alpha=0.3)
        ax_pos.plot(idx, pos, color='#22A5F7', lw=0.8)
        ax_pos.axhline(0, color='#9AA0A6', lw=0.8)
        ax_pos.set_ylabel('持仓'); ax_pos.grid(alpha=0.3)
        ax_pos.set_xlabel('时间')
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        print(f"[已保存] {out_png}")

    print(f"\n==== {os.path.basename(csv)} 常数漂移目标法(样本外自 {m['start']}) ====")
    print(f"年化 {m['annual_return']*100:6.1f}% | 波动 {m['annual_volatility']*100:5.1f}% | "
          f"夏普 {m['sharpe']:4.2f} | 回撤 {m['max_drawdown']*100:5.1f}% | "
          f"seg_std {m['seg_std']:.3f} | auto {m['autocorr']:4.2f} | linR2 {m['lin_r2']:4.2f} | "
          f"杠杆 {m['mean_leverage']:.2f}")
    return m


def rolling_vol(close, window=20):
    r = np.concatenate([[0.0], np.diff(np.log(close))])
    s = pd.Series(r)
    vol = s.shift(1).rolling(window, min_periods=1).std() * np.sqrt(PPY)
    return np.nan_to_num(vol.values, nan=np.inf)


def pick_best_k(close, decision_idx, pool=K_POOL):
    """在截止 decision_idx 的历史上选最优动量尺度 k（扣除成本后夏普最大）。"""
    h = close[:decision_idx + 1]
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


def backtest_positions(pos, close):
    r = np.concatenate([[0.0], np.diff(np.log(close))])
    turn = np.abs(np.diff(np.concatenate([[0], pos])))
    cost = turn * (FEE + SLIP)      # 调仓成本 = 换手量 × 单边费率
    net = pos * r - cost
    eq = np.cumsum(net)
    return r, net, eq, cost


def metrics(eq, ppy=PPY):
    daily = np.insert(np.diff(eq), 0, 0.0)
    n = len(eq)
    years = n / ppy
    annual_return = (eq[-1] - eq[0]) / years
    std_daily = np.std(daily) if len(daily) > 1 else 0.0
    annual_vol = std_daily * np.sqrt(ppy)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
    max_dd = np.max(-(eq - np.maximum.accumulate(eq)))
    edges = np.linspace(0, n, 13).astype(int)
    seg = [eq[edges[k + 1] - 1] - eq[edges[k] - 1] for k in range(12)]
    ac = np.corrcoef(daily[:-1], daily[1:])[0, 1] if len(daily) > 1 and np.std(daily[:-1]) > 0 else np.nan
    # 直线度：净值对时间的线性拟合 R²，越接近 1 越像缓升直线
    x = np.arange(n)
    pf = np.polyfit(x, eq, 1)
    lin = np.polyval(pf, x)
    ss_res = np.sum((eq - lin) ** 2)
    ss_tot = np.sum((eq - np.mean(eq)) ** 2)
    lin_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return dict(annual_return=annual_return, annual_volatility=annual_vol,
                sharpe=sharpe, max_drawdown=max_dd, seg_std=np.std(seg),
                autocorr=ac, lin_r2=lin_r2)


def smooth(pos, alpha=0.25, max_step=None):
    """EMA 平滑 + 可选每日最大仓变限制（消除台阶拉升，逼近缓升直线）。"""
    out = np.empty_like(pos)
    out[0] = pos[0]
    for i in range(1, len(pos)):
        t = out[i - 1] * (1 - alpha) + pos[i] * alpha
        if max_step is not None:
            t = np.clip(t, out[i - 1] - max_step, out[i - 1] + max_step)
        out[i] = t
    return out


def run(csv, train_months=24, max_leverage=2.0, vol_target=0.25, vol_window=20,
        smooth_alpha=0.25, max_step=None, out_png=None, report=True,
        bars_per_day=1, quantize=False):
    """趋势跟踪（v1）。bars_per_day: 每根K线对应多少自然日(日线=1, 30m=48)，
    用于把以"天"为单位的窗口/动量尺度换算为根(bars)，保证跨频率口径一致。
    quantize=True 时把连续仓位就近取整到整数杠杆档 {0,±1,...,±max_leverage}，
    满足真实交易杠杆必须为整数的约束。"""
    idx, close = load_ts(csv)
    n = len(close)
    bpd = bars_per_day
    ppy = int(365 * bpd)
    train_days = int(train_months * 30 * bpd)   # 训练期(根)
    month_bars = int(30 * bpd)                  # 每月重训间隔(根)
    k_pool = [max(2, int(round(k * bpd))) for k in K_POOL]  # 动量尺度按频率放大
    vol = rolling_vol(close, int(vol_window * bpd))

    pos = np.zeros(n)
    i = train_days
    while i < n:
        k = pick_best_k(close, i, pool=k_pool)
        j = i
        while j < n and (j - i) < month_bars:
            ref = close[max(0, j - k)]
            sign = 1.0 if close[j] > ref else (-1.0 if close[j] < ref else 0.0)
            v = vol[j]
            target = vol_target / v if (v > 0 and np.isfinite(v)) else 0.0
            pos[j] = np.clip(sign * target, -max_leverage, max_leverage)
            j += 1
        i = j

    pos = smooth(pos, smooth_alpha, max_step=max_step)
    if quantize:
        # 就近取整到整数杠杆 → 真实可执行仓位，且从不超过 max_leverage
        pos = np.clip(np.round(pos), -max_leverage, max_leverage)
    r, net, eq, cost = backtest_positions(pos, close)
    m = metrics(eq, ppy=ppy)
    m['mean_leverage'] = float(np.mean(np.abs(pos)))
    m['start'] = str(idx[train_days].date())
    m['eq'] = eq
    m['idx'] = idx
    m['pos'] = pos
    m['close'] = close
    m['net'] = net
    m['cost'] = cost

    if report:
        print(f"\n==== {os.path.basename(csv)} 杠杆{max_leverage} vol_target={vol_target} "
              f"(样本外自 {m['start']}) ====")
        print(f"年化 {m['annual_return']*100:6.1f}% | 波动 {m['annual_volatility']*100:5.1f}% | "
              f"夏普 {m['sharpe']:4.2f} | 回撤 {m['max_drawdown']*100:5.1f}% | "
              f"seg_std {m['seg_std']:.3f} | auto {m['autocorr']:4.2f} | linR2 {m['lin_r2']:4.2f} | "
              f"杠杆 {m['mean_leverage']:.2f}")

    if out_png:
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1, 1]})
        ax_eq, ax_dd, ax_pos = axes
        ax_eq.plot(idx, np.exp(eq), color='#3C2ECA', lw=1.5)
        ax_eq.set_title(f"{os.path.basename(csv)} · 净值(对数) 年化={m['annual_return']*100:.0f}% "
                        f"回撤={m['max_drawdown']*100:.0f}% 夏普={m['sharpe']:.2f}")
        ax_eq.set_yscale('log'); ax_eq.grid(alpha=0.3)
        dd = -(eq - np.maximum.accumulate(eq))
        ax_dd.fill_between(idx, 0, dd * 100, color='#E8463A', alpha=0.5)
        ax_dd.set_ylabel('回撤%'); ax_dd.grid(alpha=0.3)
        ax_pos.plot(idx, pos, color='#22A5F7', lw=0.8)
        ax_pos.axhline(0, color='#9AA0A6', lw=0.8)
        ax_pos.set_ylabel('持仓'); ax_pos.grid(alpha=0.3)
        ax_pos.set_xlabel('时间')
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        print(f"[已保存] {out_png}")
    return m


def scan(csv, out_png=None):
    """扫描 (杠杆, vol_target, 基线多头, smooth_alpha, max_step) 组合。
    目标：生成"接近平缓上升的直线"，无台阶拉升。
    用"长期多头偏好"避免仓位归零（消除平段→骤拉的台阶），配合平滑。"""
    idx, close = load_ts(csv)
    n = len(close)
    train_days = int(24 * 30)
    vol = rolling_vol(close, 20)
    pos_base = np.zeros(n)
    i = train_days
    while i < n:
        k = pick_best_k(close, i)
        j = i
        while j < n and (j - i) < 30:
            ref = close[max(0, j - k)]
            sign = 1.0 if close[j] > ref else (-1.0 if close[j] < ref else 0.0)
            pos_base[j] = sign
            j += 1
        i = j
    print(f"\n==== {os.path.basename(csv)} 平滑扫描（目标：平缓上升直线）====")
    print(f"{'lev':>4} {'vt':>5} {'base':>5} {'a':>5} {'step':>5} {'年化%':>7} {'夏普':>6} "
          f"{'回撤%':>6} {'seg_std':>7} {'auto':>6} {'linR2':>6}")
    best = None
    for lev in [1.0, 2.0, 3.5]:
        for vt in [0.30, 0.50, 0.80]:
            for base in [0.0, 0.4, 0.8]:
                raw = np.clip(pos_base * (vt / np.where(vol == 0, 1, vol)) + base, -lev, lev)
                for alpha in [0.35, 0.15]:
                    for step in [None, 0.15, 0.08]:
                        pos = smooth(raw, alpha, step)
                        _, _, eq, _ = backtest_positions(pos, close)
                        m = metrics(eq)
                        lin = m['lin_r2'] if np.isfinite(m['lin_r2']) else 0.0
                        auto = m['autocorr'] if np.isfinite(m['autocorr']) else 0.0
                        if m['max_drawdown'] <= 0 or not np.isfinite(auto):
                            continue
                        # 平缓为主：低seg_std + 高linR2 + 高自相关，负回撤重罚；收益只需可观
                        dd = m['max_drawdown']
                        score = -m['seg_std'] * 2.5 + lin * 1.2 + auto * 1.0 - dd * 1.0 \
                            + max(0.0, min(m['annual_return'], 0.6)) * 0.9
                        if m['annual_return'] <= 0:
                            score -= 1e6
                        if best is None or score > best[0]:
                            best = (score, lev, vt, base, alpha, step, m)
                        print(f"{lev:4.1f} {vt:5.2f} {base:5.1f} {alpha:5.2f} {str(step):>5} "
                              f"{m['annual_return']*100:7.1f} {m['sharpe']:6.2f} "
                              f"{dd*100:6.1f} {m['seg_std']:7.3f} {auto:6.2f} {lin:6.2f}")
    s, lev, vt, base, alpha, step, m = best
    print(f"\n[最优平缓] 杠杆={lev} vol_target={vt} 基线多头={base} alpha={alpha} max_step={step} -> "
          f"年化{m['annual_return']*100:.0f}% 夏普{m['sharpe']:.2f} 回撤{m['max_drawdown']*100:.0f}% "
          f"linR2={m['lin_r2']:.2f} seg_std={m['seg_std']:.3f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/ETHUSDT_1D_kline.csv")
    ap.add_argument("--train-months", type=int, default=24)
    ap.add_argument("--max-leverage", type=float, default=2.0)
    ap.add_argument("--vol-target", type=float, default=0.25)
    ap.add_argument("--vol-window", type=int, default=20)
    ap.add_argument("--smooth-alpha", type=float, default=0.25)
    ap.add_argument("--max-step", type=float, default=None)
    ap.add_argument("--target-annual", type=float, default=0.40)
    ap.add_argument("--bars-per-day", type=float, default=1.0,
                    help="每根K线对应多少自然日(日线=1, 30m=48)")
    ap.add_argument("--quantize", action="store_true",
                    help="把连续仓位就近取整到整数杠杆档")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--font", default=None, help="中文字体名如 Microsoft YaHei")
    args = ap.parse_args()
    if args.font:
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = [args.font]
        matplotlib.rcParams['axes.unicode_minus'] = False
    if args.scan:
        scan(args.csv)
    elif args.flat:
        run_flat(csv=args.csv, train_months=args.train_months,
                 target_annual=args.target_annual, vol_window=args.vol_window,
                 max_step=args.max_step, out_png=args.out,
                 bars_per_day=args.bars_per_day)
    else:
        run(csv=args.csv, train_months=args.train_months, max_leverage=args.max_leverage,
            vol_target=args.vol_target, vol_window=args.vol_window,
            smooth_alpha=args.smooth_alpha, max_step=args.max_step, out_png=args.out,
            bars_per_day=args.bars_per_day, quantize=args.quantize)


if __name__ == "__main__":
    main()