"""决策者（Portfolio Manager）。

接收市场分析报告 + 新闻 + 持仓假设检查 + 当前账户现状，输出符合
TradingDecision Schema 的具体交易举措（挂单/平仓/持有）。

关键策略要求：
- 激进风格：在保证不爆仓的前提下，杠杆建议 ≥15x（受风控上限约束）
- 每次开仓必须带止损；止盈可选
- 若存在历史持仓假设，须先检查当前行情是否偏离了原本的假设
"""
from __future__ import annotations

import json
import logging

from agents.llm import LLMClient
from agents.schemas import DECISION_JSON_SCHEMA_HINT, TradingDecision
from config import config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Portfolio Manager of a crypto perpetual futures trading desk.
You manage USDT-M perpetual futures on Binance for BTC, ETH and SOL.

Your job:
1. Read the market analysis report, news context, hypothesis check, and the current account snapshot.
2. Decide concrete trading actions per asset, output strictly as JSON matching the schema.
3. STRATEGY STYLE — AGGRESSIVE: You favor bold, high-leverage positioning. Use leverage of AT LEAST 15x
   whenever you open a position (respect the hard cap in the risk constraints). The ONLY hard limit is
   that you must NEVER risk liquidation: keep margin sufficient, set stop-loss, and size positions so
   that adverse moves to the stop-loss never blow the account.
4. MANDATORY STOP-LOSS: Every OPEN action MUST include a stop_loss. take_profit is OPTIONAL.
5. **CRITICAL — Use margin instead of quantity for OPEN actions**: For OPEN_LONG/OPEN_SHORT you MUST
   output `margin` (initial margin in USDT) instead of `quantity`. The system will automatically
   compute the order quantity = margin × leverage / entry_price for you. Set `quantity` to null.
   - `margin` is the amount of USDT margin you want to commit to this opening.
   - The resulting notional value = margin × leverage (e.g., margin=5U, leverage=15x → notional=75U).
   - You MUST ensure margin ≤ available_balance (see the account snapshot below).
   - Example: 账户可用=9.27U, 开 margin=5.0U, leverage=15x → 名义75U, 数量=75/63556≈0.00118 BTC.
   - For REPLACE_LIMIT, CANCEL_ORDERS, SET_SL_TP, CLOSE actions: still use `quantity` (coin amount).
6. Hypothesis check: If a hypothesis-check section shows you hold a position opened with an earlier
   rationale, FIRST decide whether current price action has diverged from that original hypothesis.
   - If price moved in your favor and the hypothesis still holds: HOLD or tighten SL.
   - If price broke the stop-loss level or the hypothesis is invalidated: CLOSE the position.
   - Only open NEW or additional positions on strong fresh evidence; avoid churning.
   - If no position exists, analyze from scratch (从零分析).
7. Respect all risk constraints listed in the prompt — they are hard limits enforced by the system.
8. For every OPEN action you MUST provide: margin (>0, ≤ available_balance), leverage (≥15x), and a stop_loss.
9. For every CLOSE/FLATTEN action, reason must reference current position & unrealized PnL.
10. LIMIT orders (限价挂单) are encouraged when you have a clear entry price. Prefer LIMIT in these cases:
   - You expect a pullback to a support/resistance level before the move continues (wait for a better price).
   - You want to enter only if price reaches your level (e.g. buy the dip near 1h Bollinger lower band).
   - You want to control slippage on a fast market.
   When you choose LIMIT, you MUST set "order_type": "LIMIT" and provide a concrete "price".
   Use MARKET only when you need immediate execution (e.g. breakout confirmation, stop-loss urgency).
11. ORDER MANAGEMENT (挂单管理): you CAN cancel and modify your own pending LIMIT orders.
   - CANCEL_ORDERS: cancel ALL unfilled LIMIT orders for that symbol (e.g. the price never reached
     your level, or you changed your mind). Provide only symbol + reason.
   - REPLACE_LIMIT (更改挂单): cancel all pending orders then re-place a LIMIT order at a new
     price/quantity. MUST provide "side": "BUY"/"SELL", "quantity", and "price" (new price).
   - Pending orders do NOT consume margin and do NOT affect your position; cancelling or replacing
     them never touches existing positions.
   - The account context below lists ALL your open orders with their orderId, type, side, price,
     quantity and reduceOnly flag — use them when deciding to cancel (CANCEL_ORDERS), re-place
     (REPLACE_LIMIT) or adjust stop-loss/take-profit (SET_SL_TP).
12. STOP-LOSS / TAKE-PROFIT ADJUSTMENT (随时调整持仓的止盈止损): for an EXISTING position you can
   change its protective levels at any time with "action": "SET_SL_TP".
   - Provide symbol + the new "stop_loss" and/or "take_profit" (at least ONE of them).
   - The system cancels the old protective orders and re-places them at the new prices.
   - Direction rules: for a LONG position stop_loss < current price < take_profit; for a SHORT
     position stop_loss > current price > take_profit.
13. The take_profit (止盈) is optional but recommended; the stop_loss is MANDATORY for every OPEN.
14. WAKE CONDITIONS (条件唤醒): you may set watch conditions so the system wakes you up OUTSIDE the
   normal loop when price hits your level. Output "watch_conditions" alongside instructions:
   [{"symbol": "BTCUSDT", "condition": "price_above", "value": 102000, "note": "why this level"}]
   - price_above: wake when price >= value; price_below: wake when price <= value.
   - Useful for: entering on a pullback/breakout, re-checking a stop level, monitoring risk.
   - An EMPTY list clears all previous conditions; a non-empty list REPLACES the previous set.
   - Conditions are one-shot (cleared once triggered) and auto-expire after ~24h.

Output ONLY valid JSON. No markdown fences, no extra text.
"""


def format_account_context(account, open_orders_by_symbol=None) -> str:
    """将账户现状（余额/持仓/未成交挂单）格式化为 markdown 上下文。"""
    lines = [
        "# 当前账户现状",
        "",
        f"- 账户权益(余额): {account.margin_balance:.4f} USDT",
        f"- 可用余额: {account.available_balance:.4f} USDT（本轮开仓保证金总预算，任何单笔 margin 不得超过此值）",
        f"- 未实现盈亏: {account.unrealized_pnl:+.4f} USDT",
        "",
        "## 当前持仓",
    ]
    if not account.positions:
        lines.append("（无持仓）")
    else:
        lines.append("| 币种 | 方向 | 数量 | 开仓均价 | 标记价格 | 未实现盈亏 | 杠杆 | 强平价 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for p in account.positions:
            direction = "多" if p.position_amt > 0 else "空"
            lines.append(
                f"| {p.symbol} | {direction} | {abs(p.position_amt):.6g} | "
                f"{p.entry_price:.6g} | {p.mark_price:.6g} | "
                f"{p.unrealized_pnl:+.4f} | {p.leverage:g} | "
                f"{p.liquidation_price:.6g} |"
            )
    lines.append("")
    lines.append("## 未成交挂单（open orders）")
    total = sum(len(v) for v in (open_orders_by_symbol or {}).values())
    if not total:
        lines.append("（当前无未成交挂单）")
    else:
        for sym, orders in (open_orders_by_symbol or {}).items():
            if not orders:
                continue
            lines.append(f"### {sym}")
            for o in orders:
                lines.append(
                    f"- orderId={o.get('orderId')} {o.get('side')} {o.get('type')} "
                    f"price={o.get('price')} stopPrice={o.get('stopPrice')} "
                    f"qty={o.get('origQty')} filled={o.get('executedQty')} "
                    f"reduceOnly={o.get('reduceOnly')} status={o.get('status')}"
                )
    return "\n".join(lines)


class DecisionMaker:
    """根据市场分析 + 假设检查 + 账户现状产出结构化交易决策。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def decide(
        self,
        market_report: str,
        news_context: str,
        hypothesis_context: str,
        account,
        experience_context: str = "",
        open_orders_by_symbol: dict | None = None,
    ) -> TradingDecision:
        user_content = (
            "市场分析报告：\n"
            + market_report
            + "\n\n"
            + "新闻面信息：\n"
            + news_context
            + "\n\n"
            + hypothesis_context
            + "\n\n"
            + format_account_context(account, open_orders_by_symbol)
            + "\n\n"
            + "自主经验库（历史经验，若有相关内容请务必参考）：\n"
            + (experience_context if experience_context else "（经验库为空）")
            + "\n\n"
            + "风险硬约束（必须遵守）：\n"
            + f"- 杠杆上限 {config.max_leverage}x，激进策略要求开仓杠杆 ≥ 15x（在保证金与爆仓风险可控前提下）\n"
            + f"- 单笔名义价值 ≥ {config.min_notional:.0f} USDT；名义价值 = 初始保证金 × 杠杆\n"
            + f"- 单笔开仓保证金（margin，初始保证金）≥ {config.min_margin:.2f} USDT\n"
            + f"- 单笔开仓保证金（margin）≤ 可用余额（当前 {account.available_balance:.2f} USDT），"
            + "这是系统硬底线，超出会被直接拦截\n"
            + "- 开仓时只输出 margin（初始保证金 USDT），quantity 由系统按 "
            + "数量 = margin × 杠杆 / 开仓价 自动换算，无需也不能由你给定\n"
            + "- 仓位大小（margin）由你自主决定：系统不会按比例拦截，但超可用余额会被硬拦截\n"
            + "- 已有持仓时优先检查假设是否被证伪，避免反复开平仓消耗手续费\n"
            + "- 挂单管理：未成交限价单可撤销（CANCEL_ORDERS，只给 symbol）或更改\n"
            + "  （REPLACE_LIMIT，必须给 side=BUY/SELL、quantity（币数量）、新 price）；挂单不占保证金，\n"
            + "   撤销/改单不影响已有持仓，也无需带止损\n"
            + "- 止盈止损调整：对已有持仓可用 SET_SL_TP 随时改止损/止盈（至少给一个），\n"
            + "   系统先撤销旧保护单再按新价重挂；多仓 止损<现价<止盈，空仓 止损>现价>止盈\n"
            + "\n"
            + "开仓指令字段说明：输出 margin（初始保证金 USDT，必填）、leverage、stop_loss；\n"
            + "quantity 由系统自动换算（= margin × 杠杆 / 开仓价），开仓时 quantity 置 null。\n"
            + "输出 JSON 格式（严格遵循）：\n"
            + DECISION_JSON_SCHEMA_HINT
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        data = self.llm.chat_json(
            messages,
            temperature=config.llm_temperature,
            label=f"决策者 {self.llm.model}（输出交易举措）",
        )
        try:
            decision = TradingDecision.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.error("决策 JSON 校验失败: %s\n原始: %s", exc, json.dumps(data, ensure_ascii=False)[:800])
            raise
        return decision
