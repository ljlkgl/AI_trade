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
5. Hypothesis check: If a hypothesis-check section shows you hold a position opened with an earlier
   rationale, FIRST decide whether current price action has diverged from that original hypothesis.
   - If price moved in your favor and the hypothesis still holds: HOLD or tighten SL.
   - If price broke the stop-loss level or the hypothesis is invalidated: CLOSE the position.
   - Only open NEW or additional positions on strong fresh evidence; avoid churning.
   - If no position exists, analyze from scratch (从零分析).
6. Respect all risk constraints listed in the prompt — they are hard limits enforced by the system.
7. For every OPEN action you MUST provide: quantity, leverage (≥15x), and a stop_loss.
8. For every CLOSE/FLATTEN action, reason must reference current position & unrealized PnL.
9. LIMIT orders (限价挂单) are encouraged when you have a clear entry price. Prefer LIMIT in these cases:
   - You expect a pullback to a support/resistance level before the move continues (wait for a better price).
   - You want to enter only if price reaches your level (e.g. buy the dip near 1h Bollinger lower band).
   - You want to control slippage on a fast market.
   When you choose LIMIT, you MUST set "order_type": "LIMIT" and provide a concrete "price".
   Use MARKET only when you need immediate execution (e.g. breakout confirmation, stop-loss urgency).
10. The take_profit (止盈) is optional but recommended; the stop_loss is MANDATORY for every OPEN.

Output ONLY valid JSON. No markdown fences, no extra text.
"""


def _format_account_context(account) -> str:
    """将账户现状格式化为 markdown 上下文。"""
    lines = [
        "# 当前账户现状",
        "",
        f"- 账户权益(余额): {account.margin_balance:.4f} USDT",
        f"- 可用余额: {account.available_balance:.4f} USDT",
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
            + _format_account_context(account)
            + "\n\n"
            + "自主经验库（历史经验，若有相关内容请务必参考）：\n"
            + (experience_context if experience_context else "（经验库为空）")
            + "\n\n"
            + "风险硬约束（必须遵守）：\n"
            + f"- 杠杆上限 {config.max_leverage}x，激进策略要求开仓杠杆 ≥ 15x（在保证金与爆仓风险可控前提下）\n"
            + f"- 单笔名义价值 ≥ {config.min_notional:.0f} USDT\n"
            + f"- 单笔开仓保证金 ≥ {config.min_margin:.2f} USDT（保证金 = quantity × 价格 / 杠杆）\n"
            + f"- 单笔开仓保证金 ≤ 可用余额（当前 {account.available_balance:.2f} USDT），这是系统硬底线\n"
            + "- 仓位大小（quantity）由你自主决定：综合考虑信号强度、账户余额、杠杆与止损距离；\n"
            + "  你可以在不超可用余额的前提下选择激进或保守仓位，系统不会按比例拦截\n"
            + "- 已有持仓时优先检查假设是否被证伪，避免反复开平仓消耗手续费\n"
            + "\n"
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
