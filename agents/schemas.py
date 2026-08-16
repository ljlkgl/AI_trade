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

    OPEN_LONG = "OPEN_LONG"      # 开多（买入开多仓）
    OPEN_SHORT = "OPEN_SHORT"    # 开空（卖出开空仓）
    CLOSE_LONG = "CLOSE_LONG"    # 平多（卖出平多仓）
    CLOSE_SHORT = "CLOSE_SHORT"  # 平空（买入平空仓）
    FLATTEN = "FLATTEN"          # 清仓该币种全部持仓
    HOLD = "HOLD"                # 持有不动


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
            "标的币数量（如 BTC 数量）。开仓必填；平仓可省略（缺省=全部平掉）；"
            "清仓/平仓尽量给一个具体数值，执行器会按持仓数量截断"
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

    @model_validator(mode="after")
    def _require_stop_loss_on_open(self):
        """开仓动作必须带止损；止盈可选。"""
        if self.action in (OrderAction.OPEN_LONG, OrderAction.OPEN_SHORT):
            if self.stop_loss is None:
                raise ValueError(
                    f"{self.symbol} {self.action} 必须设置 stop_loss（止损），止盈可选"
                )
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


# 输出给模型的 JSON 示例，用于 few-shot 约束格式
DECISION_JSON_SCHEMA_HINT = """{
  "market_assessment": "整体偏多/偏空/震荡的简短判断...",
  "instructions": [
    {
      "symbol": "BTCUSDT",
      "action": "OPEN_LONG",
      "order_type": "MARKET",
      "price": null,
      "quantity": 0.05,
      "leverage": 5,
      "stop_loss": 94000,
      "take_profit": 100000,
      "reason": "MACD金叉且价格站上50SMA，趋势偏多"
    }
  ],
  "risk_notes": "风险提示...（不涉及具体账户数字，只做风控说明）"
}"""


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
