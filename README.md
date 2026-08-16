# TradeTool — 币安 BTC/ETH/SOL 永续合约 AI 交易系统

提取自 [TradingAgents](../../TradingAgents) 的多 Agent 行情分析逻辑，实现为一个独立的、
可部署至服务器的币安 USDT-M 永续合约交易系统。

## 核心特性

- **多周期技术分析**（沿用 TradingAgents 指标体系）：SMA50/200、EMA10、MACD(12,26,9)、
  RSI(14)、Bollinger(20,2)、ATR(14)、VWMA(20)、MFI(14)，覆盖 15m/1h/4h/1d 周期
- **新闻面分析**（提取自 TradingAgents 的 yfinance 新闻接口）：标的新闻（BTC-USD 等）
  与宏观新闻（`yf.Search`）注入市场分析师报告
- **市场分析师 Agent**：基于指标快照 + 新闻输出结构化市场分析报告
  （趋势/动量/波动率/量能/新闻催化剂/关键位/方向偏置）
- **决策者 Agent（Portfolio Manager）**：接收「市场分析报告 + 新闻 + 持仓假设检查 +
  当前账户现状」，按指定 JSON 格式输出**具体交易举措**——开多/开空（市价或限价挂单）、
  平多/平空、清仓、持有，并给出数量、杠杆、止损、止盈与理由
- **账户现状注入**：每一轮**最先**获取账户权益、可用余额、未实现盈亏与各币种持仓
  （方向/数量/开仓均价/强平价），注入决策模型上下文
- **按余额交易**：仓位必须基于当前账户余额计算（提示词给出 `quantity = 目标保证金 × 杠杆 / 价格`），
  风控强制单笔保证金不超过可用余额
- **激进策略指导**：策略风格激进，开仓杠杆要求 **≥15x**（上限受风控硬约束），
  唯一底线是保证金充足、到止损价不爆仓
- **强制止损 + 保证金下限**：所有开仓动作**必须带 stop_loss**（Schema 与风控双层强制），
  止盈可选；单笔开仓保证金（quantity×price/leverage）**≥ MIN_MARGIN（默认 2.5 USDT）**
- **持仓假设记录与偏离检查**：开仓/调仓后把完整理由写入假设存储；下一轮决策时
  将「原始假设 vs 当前行情」注入模型，检查行情是否偏离了原本的假设
  （顺向则持有/收紧止损，触发止损或假设证伪则平仓）；账户无持仓则从零分析
- **自我反省 + 自主经验库**：每轮执行后反思者复盘决策与账户结果；模型可自主
  向经验库（`state/experience_library.json`）**写入**严重亏损教训 / 策略改进 / 风控失误等经验，
  也**有权修改或删除**已有条目；后续决策轮自动参考经验库内容
- **仓位自主决定**：仓位大小（quantity/保证金占用）由模型基于信号强度、账户余额、
  杠杆与止损距离自主决定，风控不做比例拦截；系统仅保留硬底线
- **硬风控底线**：杠杆上限、最小下单价值、最小保证金、保证金不超过可用余额、
  止损强制与方向校验，不满足的指令在执行前拦截
- **DRY_RUN 模式**：默认开启，只输出决策不真实下单，安全演练
- **测试网支持**：可一键切换币安测试网

## 目录结构

```
trade_tool/
├── main.py                  # 入口（--once / 循环模式）
├── config.py                # 配置（.env）
├── agents/
│   ├── llm.py               # OpenAI 兼容 LLM 客户端
│   ├── schemas.py           # 结构化输出 Schema（交易举措 + 反思/经验库操作）
│   ├── market_analyst.py    # 市场分析师（技术 + 新闻）
│   ├── decision_maker.py    # 决策者（激进策略 + 假设检查 + 账户现状注入）
│   └── reflector.py         # 反思者（自我反省 + 经验库维护）
└── trading/
    ├── binance_client.py    # 币安 USDT-M 合约 API 客户端
    ├── indicators.py        # 技术指标计算（TradingAgents 指标体系）
    ├── market.py            # 行情数据服务
    ├── news.py              # 新闻服务（yfinance，提取自 TradingAgents）
    ├── hypothesis.py        # 持仓假设存储 + 偏离检查上下文
    ├── experience.py        # 自主经验库（写入/修改/删除/参考）
    ├── risk.py              # 风控校验（含止损强制、保证金下限）
    ├── executor.py          # 订单执行器
    └── types.py             # 数据类型
```

## 快速开始

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置环境变量：

```bash
cp .env.example .env
# 编辑 .env：填入 LLM_API_KEY / LLM_MODEL（OpenAI 兼容接口，支持 deepseek/qwen/glm 等）
# 以及币安 API Key（测试网或主网）
```

3. 运行：

```bash
# 只执行一轮（推荐先 DRY_RUN 演练）
python main.py --once

# 循环模式（默认 60 分钟一轮）
python main.py

# 指定间隔
python main.py --interval 30
```

## 模型输出的指定格式

决策者模型必须输出如下 JSON（`agents/schemas.py` 定义）：

```json
{
  "market_assessment": "整体偏多/偏空/震荡的简短判断...",
  "instructions": [
    {
      "symbol": "BTCUSDT",
      "action": "OPEN_LONG",
      "order_type": "MARKET",
      "price": null,
      "quantity": 0.05,
      "leverage": 15,
      "stop_loss": 94000,
      "take_profit": 100000,
      "reason": "MACD金叉且价格站上50SMA，趋势偏多，新闻面无利空"
    }
  ],
  "risk_notes": "风险提示..."
}
```

`action` 可选值：

| action | 含义 | 备注 |
|---|---|---|
| `OPEN_LONG` | 开多 | **必须**带 quantity、leverage(≥15x 激进要求)、stop_loss |
| `OPEN_SHORT` | 开空 | **必须**带 quantity、leverage(≥15x 激进要求)、stop_loss |
| `CLOSE_LONG` | 平多 | 数量缺省=全部 |
| `CLOSE_SHORT` | 平空 | 数量缺省=全部 |
| `FLATTEN` | 清仓该币种 | 全部持仓平掉 |
| `HOLD` | 持有不动 | 不产生订单 |

模型同时收到**账户现状**与**持仓假设检查**：

```
# 当前账户现状
- 账户权益: 1234.56 USDT
- 可用余额: 1000.00 USDT
- 未实现盈亏: +23.45 USDT
## 当前持仓
| 币种 | 方向 | 数量 | 开仓均价 | 标记价格 | 未实现盈亏 | 杠杆 | 强平价 |
| BTCUSDT | 多 | 0.1 | 60000 | 61000 | +100 | 15 | 55000 |

# 持仓假设检查（上次开仓/调仓理由 vs 当前行情）
## BTCUSDT
- 开仓时间: 2026-08-15T10:00:00
- 方向: LONG  开仓均价: 95000  止损: 92000  止盈: 100000
- 原始开仓理由: MACD金叉趋势偏多
- 当前持仓: 数量 0.05  标记价 98000  未实现盈亏 +150.0000
- 相对开仓价偏离: +3.16%（顺向）
```

## 服务器部署

### Docker

```bash
cp .env.example .env   # 填写真实配置
docker compose up -d --build
```

容器默认以循环模式运行，`restart: unless-stopped` 保证崩溃自动重启。

### 直接部署（systemd / supervisor）

```bash
python main.py --interval 60
```

建议配合 supervisor/systemd 托管，日志输出到 stdout 由进程管理器收集。

## 安全说明

- 默认 `DRY_RUN=true`、`BINANCE_TESTNET=false`，**首次运行请务必保持 DRY_RUN 并在测试网演练**
- 确认无误后：将 `DRY_RUN` 改为 `false` 再切换主网 API Key
- 主网 Key 只授予合约交易 + 交易权限，不要开放提币权限
- 币安 API Key 存在服务器本机 .env，注意文件权限与密钥保管
- 高杠杆（≥15x）会显著放大盈亏，止损价务必合理，确保极端行情下到止损价不会爆仓

## 与 TradingAgents 的逻辑映射

| TradingAgents | 本系统 |
|---|---|
| market_analyst（get_stock_data/get_indicators） | `MarketDataService` + `MarketAnalyst` |
| news_analyst（get_news / get_global_news，yfinance） | `NewsService`（yfinance，注入市场分析师） |
| stockstats 指标（SMA/MACD/RSI/BOLL/ATR/VWMA/MFI） | `indicators.py`（纯 pandas 实现，同名指标） |
| TraderProposal（action/reasoning/entry/stop_loss/sizing） | `TradeInstruction`（动作/理由/价格/数量/杠杆/止损止盈，开仓强制止损） |
| Portfolio Manager + 账户上下文 | `DecisionMaker`（注入账户现状 + 假设检查后结构化决策） |
| TradingMemoryLog（决策复盘记忆） | `HypothesisStore`（开仓理由记录 + 偏离检查） |
| Reflector（回测结果反思） | `Reflector` + `ExperienceStore`（每轮自我反省 + 自主经验库） |
| 风控约束 | `RiskManager` 硬校验层 |

## 两档模型分工 + 三档推理强度（仿照 TradingAgents）

TradingAgents 用 `deep_think_llm` / `quick_think_llm` 两个模型按任务复杂度分工，
并用 `reasoning_effort`（low/medium/high）控制深度推理强度。本系统同样实现：

| 配置项 | 用途 | 默认 |
|---|---|---|
| `LLM_MODEL` | **quick_think_llm**：市场分析师、反思者（快速任务） | 必填 |
| `LLM_DEEP_MODEL` | **deep_think_llm**：决策者/研究经理（复杂推理）；留空则回退 `LLM_MODEL` | 空 |
| `LLM_REASONING_EFFORT` | 推理强度三档 `low` / `medium` / `high`，随请求发送；留空则不发送 | `medium` |

分工与 TradingAgents 对齐：`DecisionMaker`（对应 Research Manager / Portfolio Manager）
使用深度模型；`MarketAnalyst` 与 `Reflector` 使用快速模型。

注意：部分 OpenAI 兼容接口（如某些 SenseNova/DeepSeek 网关）不接受
`reasoning_effort` 参数，系统检测到接口报错后会自动移除该参数降级重试，不会中断运行。
