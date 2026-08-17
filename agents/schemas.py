"""结构化输出 Schema。

模型必须按以下格式输出具体交易举措（挂单、平仓、持有等），
供执行器解析后映射为币安订单。核心逻辑借鉴 TradingAgents 的
TraderProposal / PortfolioDecision：让模型直接产出结构化动作，
而非自由文本。
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class OrderAction(str, Enum):
    """交易动作。"""

    OPEN_LONG = "OPEN_LONG"              # 开多（买入开多仓）
    OPEN_SHORT = "OPEN_SHORT"            # 开空（卖出开空仓）
    CLOSE_LONG = "CLOSE_LONG"            # 平多（卖出平多仓）
    CLOSE_SHORT = "CLOSE_SHORT"          # 平空（买入平空仓）
    FLATTEN = "FLATTEN"                  # 清仓该币种全部持仓
    CANCEL_ORDERS = "CANCEL_ORDERS"      # 撤销该币种全部未成交挂单（限价单）
    REPLACE_LIMIT = "REPLACE_LIMIT"      # 更改挂单：撤销原挂单，按新价格/数量重新挂限价单
    SET_SL_TP = "SET_SL_TP"              # 调整已持仓位的止盈/止损（撤销旧保护单，按新价重挂）
    HOLD = "HOLD"                        # 持有不动


class OrderType(str, Enum):
    """订单类型。"""

    MARKET = "MARKET"   # 市价单
    LIMIT = "LIMIT"     # 限价单（挂单）


class TradeInstruction(BaseModel):
    """单条交易举措。"""

    symbol: str = Field(description="永续合约交易对，如 BTCUSDT / ETHUSDT / SOLUSDT")
    action: OrderAction = Field(description="交易动作，见 OrderAction")
    order_type: OrderType = Field(
        default=OrderType.MARKET, description="订单类型：MARKET 市价 / LIMIT 限价挂单"
    )
    price: Optional[float] = Field(
        default=None, description="LIMIT 挂单价格；MARKET 单不填"
    )
    quantity: Optional[float] = Field(
        default=None,
        description=(
            "标的币数量（如 BTC 数量）。开仓无需填写（系统按 margin×杠杆/价格自动换算）；"
            "平仓可省略（缺省=全部平掉）；REPLACE_LIMIT 更改挂单必填"
        ),
    )
    margin: Optional[float] = Field(
        default=None,
        description=(
            "初始保证金（USDT）。仅 OPEN_LONG / OPEN_SHORT 开仓必填："
            "系统按 数量 = margin × 杠杆 / 开仓价 自动换算下单，"
            "订单名义价值 = margin × 杠杆。平仓/挂单管理仍用 quantity（币数量）"
        ),
    )
    leverage: Optional[int] = Field(
        default=None, description="开仓时设置的目标杠杆倍数（1~最大杠杆）"
    )
    stop_loss: Optional[float] = Field(
        default=None, description="止损价（触发后按市价平仓，方向随仓位方向）"
    )
    take_profit: Optional[float] = Field(
        default=None, description="止盈价（触发后按市价平仓）"
    )
    reason: str = Field(description="该举措的理由，须引用具体指标或账户数据")
    side: Optional[str] = Field(
        default=None,
        description=(
            "挂单方向，仅 REPLACE_LIMIT（更改挂单）必填：BUY 挂买单 / SELL 挂卖单；"
            "其它动作忽略该字段"
        ),
    )

    @model_validator(mode="after")
    def _require_stop_loss_on_open(self):
        """开仓动作必须带止损（止盈可选）和初始保证金。"""
        if self.action in (OrderAction.OPEN_LONG, OrderAction.OPEN_SHORT):
            if self.stop_loss is None:
                raise ValueError(
                    f"{self.symbol} {self.action} 必须设置 stop_loss（止损），止盈可选"
                )
            if self.margin is None or self.margin <= 0:
                raise ValueError(
                    f"{self.symbol} {self.action} 必须设置 margin>0（初始保证金 USDT）；"
                    f"数量由系统按 margin×杠杆/价格自动换算"
                )
        return self

    @model_validator(mode="after")
    def _validate_order_management(self):
        """挂单管理动作的必填字段校验。"""
        if self.action == OrderAction.REPLACE_LIMIT:
            if self.side not in ("BUY", "SELL"):
                raise ValueError(
                    f"{self.symbol} REPLACE_LIMIT 必须设置 side=BUY/SELL（挂单方向）"
                )
            if self.price is None or self.price <= 0:
                raise ValueError(
                    f"{self.symbol} REPLACE_LIMIT 必须设置 price>0（新挂单价）"
                )
            if self.quantity is None or self.quantity <= 0:
                raise ValueError(
                    f"{self.symbol} REPLACE_LIMIT 必须设置 quantity>0（新挂单数量）"
                )
        if self.action == OrderAction.SET_SL_TP:
            if self.stop_loss is None and self.take_profit is None:
                raise ValueError(
                    f"{self.symbol} SET_SL_TP 必须提供 stop_loss 或 take_profit 至少一个"
                )
        return self


class WakeCondition(BaseModel):
    """条件唤醒（watch trigger）：价格触及阈值时唤醒系统提前分析。"""

    symbol: str = Field(description="永续合约交易对，如 BTCUSDT")
    condition: str = Field(
        description="条件类型：price_above 价格≥value 时唤醒 / price_below 价格≤value 时唤醒"
    )
    value: float = Field(description="目标价格阈值")
    note: str = Field(
        default="", description="触发原因说明（为什么关注这个价位，供触发后的分析参考）"
    )

    @model_validator(mode="after")
    def _validate(self):
        if self.condition not in ("price_above", "price_below"):
            raise ValueError(
                f"condition 必须为 price_above 或 price_below，收到: {self.condition!r}"
            )
        if self.value <= 0:
            raise ValueError(f"value 必须为正数，收到: {self.value}")
        return self


class TradingDecision(BaseModel):
    """模型输出的完整决策。"""

    market_assessment: str = Field(
        description="对当前市场整体状况（BTC/ETH/SOL）的简明判断"
    )
    instructions: list[TradeInstruction] = Field(
        description="交易举措列表。每个币种至多一条指令；无操作则该币种为 HOLD"
    )
    risk_notes: str = Field(description="风险提示与仓位/止损管理说明")
    watch_conditions: list[WakeCondition] = Field(
        default_factory=list,
        description=(
            "唤醒条件列表：正常循环外，当任一条件满足时系统会提前唤醒执行一轮分析。"
            "空列表=清除所有唤醒条件（不监控）；提供条件则全量替换上一轮的设置"
        ),
    )


# 输出给模型的 JSON 示例，用于 few-shot 约束格式
# 开仓（OPEN_LONG/OPEN_SHORT）只输出 margin（初始保证金 USDT），数量由系统自动换算；
# 挂单管理（REPLACE_LIMIT/CANCEL_ORDERS/SET_SL_TP）与平仓才用到 quantity（币数量）。
DECISION_JSON_SCHEMA_HINT = """{
  "market_assessment": "整体偏多/偏空/震荡的简短判断...",
  "instructions": [
    {
      "symbol": "BTCUSDT",
      "action": "OPEN_LONG",
      "order_type": "MARKET",
      "price": null,
      "quantity": null,
      "margin": 60,
      "leverage": 15,
      "stop_loss": 94000,
      "take_profit": 100000,
      "reason": "MACD金叉且价格站上50SMA，趋势偏多；margin=60U ≈ 名义价值 900U，约可用余额的15%"
    },
    {
      "symbol": "BTCUSDT",
      "action": "CANCEL_ORDERS",
      "order_type": "LIMIT",
      "price": null,
      "quantity": null,
      "margin": null,
      "leverage": null,
      "stop_loss": null,
      "take_profit": null,
      "reason": "原限价挂单迟迟未成交，放弃该价位计划",
      "side": null
    },
    {
      "symbol": "ETHUSDT",
      "action": "REPLACE_LIMIT",
      "order_type": "LIMIT",
      "price": 3450,
      "quantity": 2.0,
      "margin": null,
      "leverage": null,
      "stop_loss": null,
      "take_profit": null,
      "reason": "原挂单价位未触及，下移限价到支撑位重新挂单",
      "side": "BUY"
    },
    {
      "symbol": "BTCUSDT",
      "action": "SET_SL_TP",
      "order_type": "MARKET",
      "price": null,
      "quantity": null,
      "margin": null,
      "leverage": null,
      "stop_loss": 96000,
      "take_profit": 101000,
      "reason": "行情已上移，止损止盈同步上移锁定利润",
      "side": null
    }
  ],
  "watch_conditions": [
    {
      "symbol": "BTCUSDT",
      "condition": "price_above",
      "value": 102000,
      "note": "突破前高后准备追多，唤醒我复核"
    },
    {
      "symbol": "BTCUSDT",
      "condition": "price_below",
      "value": 93000,
      "note": "跌破关键支撑则风控收紧，唤醒我处理"
    }
  ],
  "risk_notes": "风险提示...（不涉及具体账户数字，只做风控说明）"
}"""


# ---------------------------------------------------------------------------
# 执行前逐条确认（Confirmer）
# ---------------------------------------------------------------------------


class ConfirmationAction(str, Enum):
    """执行前确认动作。"""

    PROCEED = "PROCEED"   # 原指令合理，照常执行
    SKIP = "SKIP"         # 指令有问题且无法可靠修正，跳过本指令
    REPLACE = "REPLACE"   # 需要调整参数（如止损/数量/价格），给出修正后的完整指令


class InstructionConfirmation(BaseModel):
    """单条指令的执行前确认结果。"""

    decision: ConfirmationAction = Field(
        description="确认决定：PROCEED 照常执行 / SKIP 跳过 / REPLACE 用修正后的指令替换执行"
    )
    instruction: Optional[TradeInstruction] = Field(
        default=None,
        description=(
            "REPLACE 时必须提供修正后的完整指令（与 TradingDecision 中相同的字段结构）；"
            "PROCEED / SKIP 时忽略"
        ),
    )
    reason: str = Field(description="确认判断理由，须引用当前价格/账户/指令的具体参数")

    @model_validator(mode="after")
    def _replace_requires_instruction(self):
        if self.decision == ConfirmationAction.REPLACE and self.instruction is None:
            raise ValueError("REPLACE 必须提供修正后的 instruction")
        return self


# ---------------------------------------------------------------------------
# 自我反省 / 自主经验库
# ---------------------------------------------------------------------------


class ExperienceAction(str, Enum):
    """经验库操作动作。"""

    WRITE = "WRITE"      # 写入新经验
    UPDATE = "UPDATE"    # 修改已有经验
    DELETE = "DELETE"    # 删除已有经验
    NONE = "NONE"        # 无操作


class ExperienceOp(BaseModel):
    """单条经验库操作。"""

    action: ExperienceAction = Field(description="经验库操作：WRITE 写入 / UPDATE 修改 / DELETE 删除 / NONE 不操作")
    experience_id: Optional[str] = Field(
        default=None, description="目标经验 id；UPDATE/DELETE 时必填，WRITE 时忽略"
    )
    category: Optional[str] = Field(
        default=None,
        description="经验分类，建议用：亏损教训 / 盈利经验 / 策略改进 / 风控失误 / 市场观察 / 执行问题",
    )
    title: Optional[str] = Field(default=None, description="经验标题；WRITE 时必填")
    content: Optional[str] = Field(
        default=None,
        description=(
            "经验正文；WRITE 时必填。应包含：发生了什么、原因分析、"
            "以后如何避免/复用（具体可执行的规则）。UPDATE 时填新的正文内容"
        ),
    )


class Reflection(BaseModel):
    """反思者的结构化输出：自我评估 + 经验库操作。"""

    self_assessment: str = Field(
        description=(
            "对本轮决策与执行的自我评估：判断依据充分吗？执行是否顺利？"
            "若有严重亏损或发现不足，诚实指出原因"
        )
    )
    severe_loss: bool = Field(
        default=False,
        description="本轮是否发生严重亏损（如单笔亏损超过账户权益的 5%，或触发止损）",
    )
    experience_ops: list[ExperienceOp] = Field(
        default_factory=list,
        description=(
            "对经验库的操作列表。严重亏损或发现不足时应 WRITE 写入经验；"
            "发现已有经验过时/错误可 UPDATE 或 DELETE。无操作时为空列表"
        ),
    )
