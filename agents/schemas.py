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
            "平仓可省略（缺省=全部平掉），也可填部分数量做部分平仓（如想平 50%，"
            "quantity 填当前持仓数量的一半）；REPLACE_LIMIT 更改挂单必填"
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
    reason: str = Field(
        description="该举措的详细理由，必须引用具体指标/价格/新闻事件时间/账户数据，说明依据与预期；不允许空泛理由",
    )
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

    @model_validator(mode="after")
    def _require_detailed_reason(self):
        """每个操作都必须写入详细理由（引用具体依据），禁止空泛理由。"""
        text = (self.reason or "").strip()
        if len(text) < 10:
            raise ValueError(
                f"{self.symbol} {self.action.value} 的 reason（理由）过于简短（{len(text)} 字符），"
                f"必须≥10 字符：详细说明依据（具体指标/价格/新闻事件时间/账户数据）与预期，"
                f"不允许「看多」「止损了」这类空泛表述"
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


class ThesisAction(str, Enum):
    """操作理由（Thesis）列表操作动作。

    模型拥有对「当前进行中的操作理由列表」的完整操作权：
    - ADD      新增一条理由记录（开新仓 / 挂新单 / 启动某个操作时），可带 parent_id 挂到父编号下
    - UPDATE   修改一条已有记录的字段（理由、备注、止盈止损、parent_id 等）
    - COMPLETE 结束一条操作：将编号标注为完成记号，系统自动级联删除该编号
               及其全部子编号关联的操作理由（如开仓=父编号、其止盈止损=各为子编号，
               层级可两层以上）
    - DELETE   删除一条记录（该操作周期已结束，如仓位被完全平掉 / 挂单撤销且不再续挂）
    - NONE     不改动列表
    """

    ADD = "ADD"
    UPDATE = "UPDATE"
    COMPLETE = "COMPLETE"
    DELETE = "DELETE"
    NONE = "NONE"


class ThesisOp(BaseModel):
    """单条「操作理由列表」操作。

    列表中的每条记录代表一个当前正在进行的操作（开仓 / 挂单 / 某个执行动作）
    及其理由。当该操作周期结束（如仓位被完全平仓）时，对应记录应被删除。
    模型拥有对列表的完整操作权；系统也会在平仓 / 撤单后自动清理过期条目，
    防止上下文过大。
    """

    action: ThesisAction = Field(
        description="列表操作：ADD 新增 / UPDATE 修改 / COMPLETE 结束（级联删除该编号及其子编号） / DELETE 删除 / NONE 不操作"
    )
    thesis_id: Optional[str] = Field(
        default=None,
        description="目标记录 id（系统在上下文中给出）。UPDATE / COMPLETE / DELETE 必填；ADD 时忽略（系统自动分配新 id）",
    )
    symbol: Optional[str] = Field(
        default=None,
        description="交易对，如 BTCUSDT；ADD 时必填，UPDATE/DELETE 时可选",
    )
    kind: Optional[str] = Field(
        default="position",
        description="操作类型：position 持仓 / limit_order 挂单 / other 其它操作；ADD 时可选，默认 position",
    )
    parent_id: Optional[str] = Field(
        default=None,
        description=(
            "父编号 id，用于建立层级关系（如 开仓=父编号、其止盈止损=各为子编号，层级可两层以上）。"
            "ADD 时可选：不填为顶层节点，填写后作为该父编号的子节点；"
            "UPDATE 时可调整所属父编号。COMPLETE 一个父编号时，其全部子孙编号会被系统级联删除"
        ),
    )
    direction: Optional[str] = Field(
        default=None, description="方向 LONG / SHORT；持仓类 ADD 时建议填写"
    )
    entry_price: Optional[float] = Field(
        default=None, description="开仓价 / 挂单价；ADD 时可选"
    )
    stop_loss: Optional[float] = Field(default=None, description="止损价；ADD 时可选")
    take_profit: Optional[float] = Field(default=None, description="止盈价；ADD 时可选")
    thesis: Optional[str] = Field(
        default=None,
        description="完整的操作理由 / 假设（为什么做这个操作、依据、预期）；ADD 时必填，UPDATE 时可更新",
    )
    note: Optional[str] = Field(
        default=None,
        description="备注（如：已实现部分止盈、行情是否仍符合预期、剩余利润空间判断等）；ADD/UPDATE 均可",
    )

    @model_validator(mode="after")
    def _validate(self):
        if self.action == ThesisAction.ADD:
            if not self.thesis or not self.symbol:
                raise ValueError("ADD 操作必须提供 symbol 与 thesis（操作理由）")
            if len(self.thesis.strip()) < 10:
                raise ValueError(
                    "ADD 操作的 thesis（操作理由）过于简短，必须≥10 字符："
                    "详细写明操作依据（具体指标/价格/新闻事件时间/账户数据）与预期"
                )
        if self.action in (
            ThesisAction.UPDATE, ThesisAction.COMPLETE, ThesisAction.DELETE,
        ) and not self.thesis_id:
            raise ValueError(
                f"{self.action.value} 操作必须提供 thesis_id（目标记录 id）"
            )
        if self.action == ThesisAction.COMPLETE and self.parent_id is not None:
            raise ValueError(
                "COMPLETE 操作不需要 parent_id（COMPLETE 父编号时系统自动级联删除其全部子孙编号）"
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
    thesis_ops: list[ThesisOp] = Field(
        default_factory=list,
        description=(
            "对「操作理由列表」的操作（可空）。你拥有该列表的完整操作权："
            "ADD 新增当前进行中操作的理由（可带 parent_id 建立父子层级，如开仓=父、其止盈止损=子，"
            "层级可两层以上）；UPDATE 更新理由/备注/parent_id；"
            "COMPLETE 将某编号标注为完成记号，系统自动级联删除该编号及其全部子编号；"
            "DELETE 删除单个编号。当某个操作周期结束（仓位被完全平掉 / 挂单撤销且不再续挂）时，"
            "务必 COMPLETE 或 DELETE 对应条目；过期条目不及时清除会导致上下文过大。无需改动时给空列表"
        ),
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
# 挂单管理（REPLACE_LIMIT/CANCEL_ORDERS/SET_SL_TP）与平仓才用到 quantity（币数量）；
# 平仓可不填 quantity（缺省=全部平掉），也可填部分数量做部分平仓（如平 50%）。
# thesis_ops 为对「操作理由列表」的操作（可空）：ADD 新增（可带 parent_id 建立父子层级，
# 如开仓=父、其止盈止损=各为子编号，层级可两层以上）/ UPDATE 修改 / COMPLETE 结束（标注完成记号，
# 系统自动级联删除该编号及其全部子编号）/ DELETE 删除单个编号。
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
      "action": "CLOSE_LONG",
      "order_type": "MARKET",
      "price": null,
      "quantity": 0.05,
      "margin": null,
      "leverage": null,
      "stop_loss": null,
      "take_profit": null,
      "reason": "已到止盈位，先平掉一半（0.05 BTC）落袋，剩余持仓继续持有",
      "side": null
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
  "thesis_ops": [
    {
      "action": "ADD",
      "thesis_id": null,
      "symbol": "BTCUSDT",
      "kind": "position",
      "parent_id": null,
      "direction": "LONG",
      "entry_price": 95000,
      "stop_loss": 94000,
      "take_profit": 100000,
      "thesis": "ETF 资金持续流入+MACD金叉，回调后重启上行，先平一半后剩单持有",
      "note": "已部分止盈 50%，剩余仓位继续持有，行情未偏离预期"
    },
    {
      "action": "ADD",
      "thesis_id": null,
      "symbol": "BTCUSDT",
      "kind": "other",
      "parent_id": "th_20260817_....",
      "direction": null,
      "entry_price": null,
      "stop_loss": 94000,
      "take_profit": null,
      "thesis": "止损保护子编号（挂在开仓父编号下，作为其子节点）",
      "note": null
    },
    {
      "action": "COMPLETE",
      "thesis_id": "th_20260817_....",
      "symbol": null,
      "kind": null,
      "parent_id": null,
      "direction": null,
      "entry_price": null,
      "stop_loss": null,
      "take_profit": null,
      "thesis": null,
      "note": null
    },
    {
      "action": "DELETE",
      "thesis_id": "th_20260817_....",
      "symbol": null,
      "kind": null,
      "parent_id": null,
      "direction": null,
      "entry_price": null,
      "stop_loss": null,
      "take_profit": null,
      "thesis": null,
      "note": null
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
    reason: str = Field(
        description="确认判断理由，须引用当前价格/账户/指令的具体参数，禁止空泛理由"
    )

    @model_validator(mode="after")
    def _replace_requires_instruction(self):
        if self.decision == ConfirmationAction.REPLACE and self.instruction is None:
            raise ValueError("REPLACE 必须提供修正后的 instruction")
        text = (self.reason or "").strip()
        if len(text) < 10:
            raise ValueError(
                f"确认理由过于简短（{len(text)} 字符），必须≥10 字符："
                "引用当前价格/账户/指令具体参数说明判断依据"
            )
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
