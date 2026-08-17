"""市场分析师。

借鉴 TradingAgents 的 market_analyst + news_analyst 提示词逻辑：
基于技术指标（SMA/EMA/MACD/RSI/BOLL/ATR/VWMA/MFI）与新闻面信息，
输出详细、可执行的市场分析报告，供决策者参考。
"""
from __future__ import annotations

import logging

from agents.llm import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a professional crypto perpetual futures market analyst.
You will be given technical indicator snapshots across multiple timeframes
(15m / 1h / 4h / 1d) for BTC, ETH and SOL perpetual contracts, plus recent news.

Write a detailed, nuanced market analysis report covering:
1. Overall trend assessment for each asset (uptrend / downtrend / range) with evidence from SMA50/200, EMA10, price vs MA.
2. Momentum: MACD (cross, histogram), RSI (overbought/oversold), MFI (money flow).
3. Volatility: Bollinger bands (position vs bands, squeeze/expansion), ATR magnitude.
4. Volume confirmation: VWMA relationship with price.
5. News catalysts: For each news item, explicitly state:
   - Event time (when the event happened or was published, e.g., "2026-08-16 09:30 UTC")
   - Impact direction (BULLISH / BEARISH / NEUTRAL) — state which asset it affects and why
   - Price-in assessment: Is this event already fully reflected in the current price? Choose one of:
     "Not priced in" / "Partially priced in" / "Fully priced in"
   - Remaining profit potential: If not fully priced in, estimate the remaining upside or downside
     space (e.g., "1-2% upside remaining", "0.5-1% downside still to materialize").
   - If the news is noise or too old to matter, say so explicitly. Do NOT invent impact where none exists.
6. For each asset, a clear directional bias: LONG / SHORT / NEUTRAL, with a confidence level (high/medium/low) and the main reason.
7. Cross-asset relative strength: which of BTC/ETH/SOL looks strongest/weakest and why.

Be specific and cite actual indicator values and news headlines. Do NOT invent data not present.
End with a concise markdown table summarizing per-asset bias and key signals.
"""


class MarketAnalyst:
    """市场技术分析师（含新闻面）。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def analyze(self, market_context: str, news_context: str = "") -> str:
        from datetime import datetime, timezone

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        user_parts = [
            f"当前时间（UTC）：{now_utc}\n\n",
            "以下是 BTC/ETH/SOL 永续合约的多周期技术指标快照，请基于此输出分析报告：\n\n",
            market_context,
        ]
        if news_context and news_context.strip():
            user_parts.extend(["\n\n新闻面信息（供基本面/事件驱动分析参考）：\n\n", news_context])
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "".join(user_parts)},
        ]
        return self.llm.chat(messages, temperature=0.3, label=f"市场分析师 {self.llm.model}")
