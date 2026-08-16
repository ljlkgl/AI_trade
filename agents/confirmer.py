"""执行前确认者（Confirmer）。

对应需求：模型输出完整 JSON 决策后，每执行一条指令前再调用模型确认一次，
确认过程中模型也可输出新动作（REPLACE 修正指令），以便及时调整参数错误。

使用 quick_think_llm（快速模型）：确认是单条指令的轻量复核，决策主体仍在
决策者（deep 模型）完成；本环节只做执行前的二次把关。
"""
from __future__ import annotations

import json
import logging

from agents.llm import LLMClient
from agents.schemas import InstructionConfirmation, TradeInstruction
from config import config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Confirmer of a crypto perpetual futures trading desk.
The Portfolio Manager has already produced the full decision JSON. Your ONLY job is a quick
execution-time review of EACH instruction right before it is actually placed on Binance.

Goal: catch parameter mistakes (price, quantity, direction, stop-loss/take-profit too close to
current price, margin vs available balance, etc.) and correct them in time.

Decide ONE of:
1. PROCEED: the instruction is reasonable as-is — execute it unchanged. (default when no clear problem)
2. SKIP: the instruction has a problem you cannot reliably fix — skip it this round.
3. REPLACE: the instruction needs parameter fixes — provide the corrected COMPLETE instruction
   (same fields as a TradingDecision instruction). Common fixes: adjust stop_loss further from
   current price, reduce quantity, correct a wrong direction, fix a LIMIT price.

Rules:
- Do NOT modify instructions without a concrete reason tied to the data you see (price/account).
  When in doubt, PROCEED.
- For any OPEN/REPLACE action keep the hard constraints: quantity>0, leverage within cap,
  stop_loss mandatory & on the correct side (LONG: sl < price < tp; SHORT: sl > price > tp).
- For REPLACE you MUST output the full corrected instruction JSON.

Output ONLY valid JSON:
{
  "decision": "PROCEED" | "SKIP" | "REPLACE",
  "instruction": null | { ...corrected TradeInstruction... },
  "reason": "brief justification referencing concrete numbers"
}
No markdown fences, no extra text.
"""


class Confirmer:
    """单条指令执行前确认。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def confirm(
        self,
        instruction: TradeInstruction,
        mark_price: float,
        account_context: str,
        prior_exec_summary: str,
        market_assessment: str = "",
    ) -> InstructionConfirmation:
        """确认一条指令，返回决定（可含替换指令）。"""
        user_parts = [
            "待确认指令（即将执行，请复核）：\n",
            json.dumps(instruction.model_dump(), ensure_ascii=False, indent=2),
            f"\n\n该币种当前标记价格: {mark_price:.6g}",
            "\n\n最新账户现状：\n",
            account_context,
        ]
        if prior_exec_summary and prior_exec_summary.strip():
            user_parts.extend(["\n\n本轮已执行指令结果（供参考是否重复/冲突）：\n", prior_exec_summary])
        if market_assessment:
            user_parts.extend(["\n\n决策者市场判断：\n", market_assessment[:300]])
        user_parts.append(
            "\n\n请输出确认决定（PROCEED / SKIP / REPLACE）。"
            "若需 REPLACE，请给出修正后的完整指令 JSON。"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "".join(user_parts)},
        ]
        data = self.llm.chat_json(
            messages,
            temperature=config.llm_temperature,
            label=f"确认者 {self.llm.model}（复核 {instruction.symbol} {instruction.action.value}）",
        )
        try:
            return InstructionConfirmation.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "确认 JSON 校验失败: %s\n原始: %s",
                exc, json.dumps(data, ensure_ascii=False)[:800],
            )
            raise
