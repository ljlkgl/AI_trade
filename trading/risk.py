"""风控模块：在执行下单前对指令做硬性校验。

校验规则（与决策者提示词中的硬约束一致）：
1. 杠杆 ≤ max_leverage
2. 单笔名义价值 ≥ min_notional；单笔保证金 ≥ min_margin
3. 单笔保证金 ≤ 账户可用余额（真实可成交性底线）
4. 止损强制与方向合理性
仓位大小（quantity/保证金占用）由模型自主决定，不做比例拦截。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from agents.schemas import OrderAction, TradeInstruction
from config import config
from trading.types import AccountInfo, Position, SymbolInfo

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    ok: bool
    errors: list[str]


class RiskManager:
    """风控校验器。"""

    def __init__(
        self,
        max_position_ratio: Optional[float] = None,
        max_total_position_ratio: Optional[float] = None,
        max_leverage: Optional[int] = None,
        min_notional: Optional[float] = None,
        min_margin: Optional[float] = None,
    ) -> None:
        self.max_position_ratio = (
            max_position_ratio or config.max_position_ratio
        )
        self.max_total_position_ratio = (
            max_total_position_ratio or config.max_total_position_ratio
        )
        self.max_leverage = max_leverage or config.max_leverage
        self.min_notional = min_notional or config.min_notional
        self.min_margin = min_margin if min_margin is not None else config.min_margin

    # ---------- 辅助 ----------

    def current_position(
        self, account: AccountInfo, symbol: str
    ) -> Optional[Position]:
        for p in account.positions:
            if p.symbol == symbol:
                return p
        return None

    # ---------- 指令校验 ----------

    def validate_instruction(
        self,
        instruction: TradeInstruction,
        account: AccountInfo,
        symbol_info: SymbolInfo,
        mark_price: float,
    ) -> RiskCheckResult:
        """校验单条指令，返回结果。"""
        errors: list[str] = []
        equity = account.margin_balance if account.margin_balance > 0 else account.total_balance

        if instruction.action == OrderAction.HOLD:
            return RiskCheckResult(ok=True, errors=[])

        pos = self.current_position(account, instruction.symbol)

        # 开仓类：校验杠杆 + 名义价值 + 保证金（仓位大小交给模型自主决定）
        if instruction.action in (OrderAction.OPEN_LONG, OrderAction.OPEN_SHORT):
            qty = instruction.quantity
            lev = instruction.leverage or 1
            if qty is None or qty <= 0:
                errors.append(f"{instruction.symbol}: 开仓必须提供 quantity>0")
            else:
                new_nominal = abs(qty) * (instruction.price or mark_price)
                if new_nominal < self.min_notional:
                    errors.append(
                        f"{instruction.symbol}: 单笔名义价值 {new_nominal:.2f} "
                        f"低于最小限制 {self.min_notional:.2f} USDT"
                    )
            if lev > self.max_leverage:
                errors.append(
                    f"{instruction.symbol}: 杠杆 {lev} 超过上限 {self.max_leverage}"
                )
            # 保证金下限校验：保证金 = quantity * price / leverage，须 ≥ min_margin
            if qty is not None and qty > 0:
                margin = abs(qty) * (instruction.price or mark_price) / lev
                if margin < self.min_margin:
                    errors.append(
                        f"{instruction.symbol}: 开仓保证金 {margin:.4f} USDT "
                        f"低于最小保证金 {self.min_margin:.2f} USDT（quantity×price/leverage）"
                    )
                # 可用余额校验：单笔保证金不得超过账户可用余额（真实可成交性底线）
                if margin > account.available_balance:
                    errors.append(
                        f"{instruction.symbol}: 开仓保证金 {margin:.4f} USDT "
                        f"超过可用余额 {account.available_balance:.4f} USDT"
                    )
            # 止损强制：开仓必须带止损
            if instruction.stop_loss is None:
                errors.append(f"{instruction.symbol}: 开仓必须设置 stop_loss（止损）")
            else:
                if instruction.action == OrderAction.OPEN_LONG and instruction.stop_loss >= (instruction.price or mark_price):
                    errors.append(f"{instruction.symbol}: 多单止损价需低于开仓价")
                if instruction.action == OrderAction.OPEN_SHORT and instruction.stop_loss <= (instruction.price or mark_price):
                    errors.append(f"{instruction.symbol}: 空单止损价需高于开仓价")

        # 平仓类：校验数量不超过持仓
        if instruction.action in (OrderAction.CLOSE_LONG, OrderAction.CLOSE_SHORT, OrderAction.FLATTEN):
            if pos is None:
                errors.append(f"{instruction.symbol}: 无持仓，无需平仓")
            else:
                side_ok = (
                    instruction.action == OrderAction.FLATTEN
                    or (instruction.action == OrderAction.CLOSE_LONG and pos.position_amt > 0)
                    or (instruction.action == OrderAction.CLOSE_SHORT and pos.position_amt < 0)
                )
                if not side_ok:
                    errors.append(
                        f"{instruction.symbol}: 平仓方向与持仓方向不一致（持仓 {pos.position_amt:+.6g}）"
                    )
                if instruction.quantity is not None and abs(instruction.quantity) > abs(pos.position_amt):
                    errors.append(
                        f"{instruction.symbol}: 平仓数量 {instruction.quantity} 超过持仓 {abs(pos.position_amt):.6g}"
                    )

        return RiskCheckResult(ok=len(errors) == 0, errors=errors)

    def validate_decision(
        self,
        instructions: list[TradeInstruction],
        account: AccountInfo,
        price_map: dict[str, float],
        symbol_info_map: dict[str, SymbolInfo],
    ) -> tuple[list[TradeInstruction], list[RiskCheckResult]]:
        """校验整个决策。返回 (通过的指令, 校验结果列表)。"""
        passed: list[TradeInstruction] = []
        results: list[RiskCheckResult] = []
        for ins in instructions:
            mark = price_map.get(ins.symbol, 0.0)
            info = symbol_info_map.get(ins.symbol)
            if mark <= 0 or info is None:
                results.append(
                    RiskCheckResult(ok=False, errors=[f"{ins.symbol}: 缺少标记价格或交易对信息"])
                )
                continue
            res = self.validate_instruction(ins, account, info, mark)
            results.append(res)
            if res.ok:
                passed.append(ins)
            else:
                logger.warning(
                    "指令被风控拦截 %s %s: %s",
                    ins.symbol, ins.action, "; ".join(res.errors),
                )
        return passed, results
