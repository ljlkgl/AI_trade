"""订单执行器：将 TradingDecision 中的指令映射为币安订单。

处理开仓（设置杠杆/保证金模式→下单→挂止损止盈）、平仓（reduceOnly）。
DRY_RUN 模式下只打印不真实下单。
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from agents.schemas import OrderAction, OrderType, TradeInstruction
from config import config
from trading.binance_client import BinanceClient, BinanceError
from trading.risk import RiskManager
from trading.types import AccountInfo

logger = logging.getLogger(__name__)

# 止盈止损单使用的 reduceOnly 标志位常量
STOP_ORDER_TIF = "GTC"


def round_to_step(value: float, step: float) -> float:
    """按 stepSize 向下取整，避免精度错误。"""
    if step <= 0:
        return value
    n = int(round(value / step))
    return n * step


def round_price(value: float, tick: float) -> float:
    if tick <= 0:
        return value
    n = int(round(value / tick))
    return n * tick


class OrderExecutor:
    """执行交易决策。"""

    def __init__(
        self,
        client: BinanceClient,
        risk_manager: RiskManager,
        dry_run: Optional[bool] = None,
    ) -> None:
        self.client = client
        self.risk = risk_manager
        self.dry_run = config.dry_run if dry_run is None else dry_run

    # ---------- 工具 ----------

    def _quantity(self, symbol_info, qty: float) -> float:
        q = round_to_step(abs(qty), symbol_info.qty_step)
        return float(q)

    def _price(self, symbol_info, price: float) -> float:
        return float(round_price(price, symbol_info.price_tick))

    def _client_order_id(self, tag: str) -> str:
        return f"{tag}{uuid.uuid4().hex[:12]}"

    # ---------- 核心执行 ----------

    def execute(
        self,
        instructions: list[TradeInstruction],
        account: AccountInfo,
    ) -> list[dict]:
        """执行指令列表，返回执行结果日志列表。"""
        results: list[dict] = []
        for ins in instructions:
            try:
                result = self._execute_one(ins, account)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.exception("执行指令失败 %s %s: %s", ins.symbol, ins.action, exc)
                results.append(
                    {
                        "symbol": ins.symbol,
                        "action": ins.action.value,
                        "status": "FAILED",
                        "error": str(exc),
                    }
                )
        return results

    def _execute_one(self, ins: TradeInstruction, account: AccountInfo) -> dict:
        symbol_info = self.client.get_symbol_info(ins.symbol)
        mark_price = self.client.get_ticker_price(ins.symbol)
        pos = next((p for p in account.positions if p.symbol == ins.symbol), None)

        base = {
            "symbol": ins.symbol,
            "action": ins.action.value,
            "dry_run": self.dry_run,
        }

        if ins.action == OrderAction.HOLD:
            base.update(status="SKIPPED", detail="持有不动")
            return base

        if ins.action in (OrderAction.OPEN_LONG, OrderAction.OPEN_SHORT):
            return self._open(ins, symbol_info, mark_price, base)
        if ins.action in (OrderAction.CLOSE_LONG, OrderAction.CLOSE_SHORT):
            return self._close(ins, pos, mark_price, base)
        if ins.action == OrderAction.FLATTEN:
            return self._flatten(ins, pos, mark_price, base)
        raise ValueError(f"未知动作: {ins.action}")

    def _open(
        self, ins: TradeInstruction, symbol_info, mark_price: float, base: dict
    ) -> dict:
        side = "BUY" if ins.action == OrderAction.OPEN_LONG else "SELL"
        qty = self._quantity(symbol_info, ins.quantity or 0)
        if qty <= 0:
            base.update(status="REJECTED", error="数量无效")
            return base
        price = None
        order_type = "MARKET"
        if ins.order_type == OrderType.LIMIT:
            if not ins.price:
                base.update(status="REJECTED", error="限价单缺少价格")
                return base
            order_type = "LIMIT"
            price = self._price(symbol_info, ins.price)

        if self.dry_run:
            logger.info(
                "[DRY_RUN] %s %s qty=%s price=%s lev=%s sl=%s tp=%s",
                ins.symbol, ins.action.value, qty, price, ins.leverage,
                ins.stop_loss, ins.take_profit,
            )
            base.update(
                status="DRY_RUN",
                order_type=order_type,
                side=side,
                quantity=qty,
                price=price,
            )
            return base

        # 真实下单
        leverage = ins.leverage if ins.leverage else 1
        try:
            self.client.set_margin_type(ins.symbol, "ISOLATED")
        except BinanceError as exc:
            logger.warning("设置保证金模式失败(可能已是ISOLATED): %s", exc)
        try:
            self.client.set_leverage(ins.symbol, leverage)
        except BinanceError as exc:
            logger.warning("设置杠杆失败: %s", exc)

        order = self.client.place_order(
            symbol=ins.symbol,
            side=side,
            order_type=order_type,
            quantity=qty,
            price=price,
            time_in_force="GTC" if order_type == "LIMIT" else None,
            client_order_id=self._client_order_id("open"),
        )
        result = dict(base)
        result.update(
            status="OPENED",
            order=order,
            quantity=qty,
            price=price,
            order_type=order_type,
        )
        # 挂止损止盈（仅市价开仓成功后挂；限价单也可挂，但注意订单可能未成交）
        if ins.stop_loss is not None or ins.take_profit is not None:
            sl_tp = self._place_stop_loss_take_profit(ins, qty)
            result["sl_tp_orders"] = sl_tp
        return result

    def _place_stop_loss_take_profit(self, ins: TradeInstruction, qty: float) -> list[dict]:
        """为仓位挂 STOP_MARKET / TAKE_PROFIT_MARKET 保护单。"""
        placed = []
        side = "SELL" if ins.action == OrderAction.OPEN_LONG else "BUY"
        symbol_info = self.client.get_symbol_info(ins.symbol)
        if ins.stop_loss is not None:
            sl = self._price(symbol_info, ins.stop_loss)
            order = self.client.place_order(
                symbol=ins.symbol,
                side=side,
                order_type="STOP_MARKET",
                quantity=qty,
                stop_price=sl,
                reduce_only=True,
                time_in_force=STOP_ORDER_TIF,
                client_order_id=self._client_order_id("sl"),
            )
            placed.append({"kind": "stop_loss", "order": order})
        if ins.take_profit is not None:
            tp = self._price(symbol_info, ins.take_profit)
            order = self.client.place_order(
                symbol=ins.symbol,
                side=side,
                order_type="TAKE_PROFIT_MARKET",
                quantity=qty,
                stop_price=tp,
                reduce_only=True,
                time_in_force=STOP_ORDER_TIF,
                client_order_id=self._client_order_id("tp"),
            )
            placed.append({"kind": "take_profit", "order": order})
        return placed

    def _close(
        self, ins: TradeInstruction, pos, mark_price: float, base: dict
    ) -> dict:
        if pos is None:
            base.update(status="REJECTED", error="无持仓")
            return base
        if ins.action == OrderAction.CLOSE_LONG and pos.position_amt <= 0:
            base.update(status="REJECTED", error="无多仓")
            return base
        if ins.action == OrderAction.CLOSE_SHORT and pos.position_amt >= 0:
            base.update(status="REJECTED", error="无空仓")
            return base

        close_side = "SELL" if pos.position_amt > 0 else "BUY"
        qty = self._quantity_for_close(ins, pos)
        return self._do_close(ins, pos, close_side, qty, base)

    def _flatten(self, ins: TradeInstruction, pos, mark_price: float, base: dict) -> dict:
        if pos is None:
            base.update(status="REJECTED", error="无持仓")
            return base
        close_side = "SELL" if pos.position_amt > 0 else "BUY"
        qty = abs(pos.position_amt)
        return self._do_close(ins, pos, close_side, qty, base)

    def _quantity_for_close(self, ins: TradeInstruction, pos) -> float:
        """平仓数量：优先指令值，缺省为全部持仓。"""
        if ins.quantity is not None and ins.quantity > 0:
            return min(abs(ins.quantity), abs(pos.position_amt))
        return abs(pos.position_amt)

    def _do_close(self, ins: TradeInstruction, pos, close_side: str, qty: float, base: dict) -> dict:
        symbol_info = self.client.get_symbol_info(ins.symbol)
        qty = self._quantity(symbol_info, qty)
        if qty <= 0:
            base.update(status="REJECTED", error="平仓数量无效")
            return base

        order_type = "MARKET"
        price = None
        if ins.order_type == OrderType.LIMIT and ins.price:
            order_type = "LIMIT"
            price = self._price(symbol_info, ins.price)

        if self.dry_run:
            logger.info(
                "[DRY_RUN] %s %s qty=%s side=%s", ins.symbol, ins.action.value, qty, close_side
            )
            base.update(status="DRY_RUN", side=close_side, quantity=qty, order_type=order_type, price=price)
            return base

        order = self.client.close_position(
            symbol=ins.symbol,
            side=close_side,
            quantity=qty,
            order_type=order_type,
            price=price,
        )
        base.update(status="CLOSED", side=close_side, quantity=qty, order=order)
        return base
