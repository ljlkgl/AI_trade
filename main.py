"""交易系统主入口。

用法：
  python main.py            # 循环模式（按 INTERVAL_MINUTES 轮询）
  python main.py --once     # 只执行一轮
  python main.py --interval 30   # 覆盖轮询间隔（分钟）

每一轮流程（顺序固定）：
1. 最先获取账户现状（余额/持仓/未实现盈亏）
2. 若账户无任何持仓 → 从零分析（不携带历史假设）；否则加载持仓假设检查上下文
3. 拉取行情 + 指标 + 新闻 → 市场分析师报告
4. 决策者（接收市场报告 + 新闻 + 假设检查 + 经验库 + 账户现状）→ 结构化举措
5. 风控校验 → 执行
6. 若发生开仓/调仓：将完整理由记录到假设存储，供下一轮检查行情是否偏离
7. 反思者复盘本轮结果，自主写入/修改/删除经验库条目，供以后参考
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from typing import Optional

from agents.decision_maker import DecisionMaker
from agents.llm import LLMClient
from agents.market_analyst import MarketAnalyst
from agents.reflector import Reflector
from agents.schemas import ExperienceAction
from config import config
from trading.binance_client import BinanceClient
from trading.experience import ExperienceStore
from trading.executor import OrderExecutor
from trading.hypothesis import HypothesisStore, build_hypothesis_check_context
from trading.market import MarketDataService
from trading.news import NewsService
from trading.risk import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trading_system")


def _format_account_ctx(account) -> str:
    """将账户现状格式化为反思者用的 markdown 上下文。"""
    lines = [
        f"- 账户权益: {account.margin_balance:.4f} USDT",
        f"- 可用余额: {account.available_balance:.4f} USDT",
        f"- 未实现盈亏: {account.unrealized_pnl:+.4f} USDT",
    ]
    for p in account.positions:
        direction = "多" if p.position_amt > 0 else "空"
        lines.append(
            f"- 持仓 {p.symbol} {direction} qty={abs(p.position_amt):.6g} "
            f"entry={p.entry_price:.6g} mark={p.mark_price:.6g} "
            f"pnl={p.unrealized_pnl:+.4f} lev={p.leverage:g}"
        )
    return "\n".join(lines)


class TradingSystem:
    """整合行情→分析→决策→风控→执行→假设记录的主流程。"""

    def __init__(self, interval_minutes: Optional[int] = None) -> None:
        self.interval_minutes = interval_minutes or config.interval_minutes
        self.client = BinanceClient(
            api_key=config.binance_api_key,
            api_secret=config.binance_api_secret,
            testnet=config.binance_testnet,
        )
        # 两档模型分工（仿照 TradingAgents 的 deep/quick 分工）：
        # - quick_think_llm（LLM_MODEL）：市场分析师、反思者 —— 快速任务
        # - deep_think_llm（LLM_DEEP_MODEL）：决策者/研究经理 —— 复杂推理
        self.llm = LLMClient()
        self.llm_deep = (
            LLMClient(model=config.llm_deep_model) if config.llm_deep_model else self.llm
        )
        self.market_data = MarketDataService(self.client)
        self.news_service = NewsService()
        self.market_analyst = MarketAnalyst(self.llm)
        self.decision_maker = DecisionMaker(self.llm_deep)
        self.reflector = Reflector(self.llm)
        self.risk = RiskManager()
        self.executor = OrderExecutor(self.client, self.risk)
        self.hypotheses = HypothesisStore()
        self.experiences = ExperienceStore()
        self.symbols = config.symbols
        self._last_reflection: Optional[dict] = None

    def run_once(self) -> dict:
        """执行一轮完整的 行情→分析→决策→风控→执行→假设记录。"""
        logger.info("===== 开始一轮交易决策 @ %s =====", datetime.now().isoformat())
        result: dict = {
            "timestamp": datetime.now().isoformat(),
            "symbols": self.symbols,
            "testnet": config.binance_testnet,
            "dry_run": config.dry_run,
        }

        # 1. 最先获取账户现状（决策者必需）
        try:
            account = self.client.get_account()
            logger.info(
                "账户: 权益=%.2f 可用=%.2f 未实现盈亏=%.2f 持仓数=%d",
                account.margin_balance, account.available_balance,
                account.unrealized_pnl, len(account.positions),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("获取账户失败: %s", exc)
            result["error"] = f"账户获取失败: {exc}"
            return result
        result["account"] = {
            "margin_balance": account.margin_balance,
            "available_balance": account.available_balance,
            "unrealized_pnl": account.unrealized_pnl,
            "positions": [
                {"symbol": p.symbol, "side": "LONG" if p.position_amt > 0 else "SHORT",
                 "qty": abs(p.position_amt), "entry": p.entry_price,
                 "mark": p.mark_price, "pnl": p.unrealized_pnl,
                 "leverage": p.leverage, "liq": p.liquidation_price}
                for p in account.positions
            ],
        }

        # 2. 假设检查上下文：无仓位则从零分析，并清理失效假设
        account_positions = {p.symbol: p for p in account.positions}
        for sym in self.symbols:
            if sym not in account_positions and self.hypotheses.get(sym):
                logger.info("%s 已无持仓，清除历史假设（从零分析）", sym)
                self.hypotheses.remove(sym)
        hypothesis_context = build_hypothesis_check_context(
            self.hypotheses, account_positions
        )
        result["hypothesis_context"] = hypothesis_context

        # 3. 市场上下文（多币种多周期）+ 新闻
        try:
            market_context = self.market_data.build_market_context_for_symbols(
                self.symbols, limit=config.klines_limit
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("获取行情失败: %s", exc)
            result["error"] = f"行情获取失败: {exc}"
            return result
        result["market_context_len"] = len(market_context)

        try:
            news_context = self.news_service.build_news_context(self.symbols)
        except Exception as exc:  # noqa: BLE001
            logger.warning("获取新闻失败（继续分析）: %s", exc)
            news_context = "（新闻获取失败，本轮忽略新闻面）"
        result["news_context_len"] = len(news_context)

        # 4. 市场分析师产出报告（技术 + 新闻）
        try:
            market_report = self.market_analyst.analyze(market_context, news_context)
            logger.info("市场分析报告已生成（%d 字符）", len(market_report))
        except Exception as exc:  # noqa: BLE001
            logger.error("市场分析失败: %s", exc)
            result["error"] = f"市场分析失败: {exc}"
            return result
        result["market_report"] = market_report

        # 5. 决策者输出结构化举措（含账户现状 + 假设检查 + 经验库）
        try:
            experience_context = self.experiences.format_for_context()
            decision = self.decision_maker.decide(
                market_report, news_context, hypothesis_context, account,
                experience_context=experience_context,
            )
            logger.info(
                "决策: %d 条指令, 评估=%s",
                len(decision.instructions),
                decision.market_assessment[:120],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("决策失败: %s", exc)
            result["error"] = f"决策失败: {exc}"
            return result
        result["market_assessment"] = decision.market_assessment
        result["risk_notes"] = decision.risk_notes

        # 6. 风控校验
        price_map: dict[str, float] = {}
        symbol_info_map = {}
        for sym in self.symbols:
            try:
                price_map[sym] = self.client.get_ticker_price(sym)
                symbol_info_map[sym] = self.client.get_symbol_info(sym)
            except Exception as exc:  # noqa: BLE001
                logger.warning("获取 %s 价格/精度失败: %s", sym, exc)

        passed, risk_results = self.risk.validate_decision(
            decision.instructions, account, price_map, symbol_info_map
        )
        result["risk_blocked"] = [
            {"symbol": r.errors[0].split(":")[0], "errors": r.errors}
            for r in risk_results if not r.ok
        ]
        result["instructions_after_risk"] = [
            {
                "symbol": i.symbol, "action": i.action.value,
                "order_type": i.order_type.value, "price": i.price,
                "quantity": i.quantity, "leverage": i.leverage,
                "stop_loss": i.stop_loss, "take_profit": i.take_profit,
                "reason": i.reason,
            }
            for i in passed
        ]

        # 7. 执行
        if passed:
            exec_results = self.executor.execute(passed, account)
            result["execution"] = exec_results
            for r in exec_results:
                logger.info(
                    "执行结果 %s %s -> %s%s",
                    r.get("symbol"), r.get("action"), r.get("status"),
                    " (DRY_RUN)" if config.dry_run else "",
                )
        else:
            logger.info("无通过风控的指令，本轮不交易")
            exec_results = []
            result["execution"] = []

        # 8. 记录开仓/调仓理由到假设存储（供下一轮检查行情是否偏离）
        self._record_hypotheses(passed, exec_results, account, decision, price_map)

        # 9. 反思者复盘本轮，自主维护经验库（写入/修改/删除）
        self._reflect_and_apply(
            decision, exec_results, account, hypothesis_context, market_report,
        )
        if self._last_reflection:
            result["reflection"] = self._last_reflection

        logger.info("===== 本轮结束 =====")
        return result

    def _reflect_and_apply(
        self,
        decision,
        exec_results: list[dict],
        account,
        hypothesis_context: str,
        market_report: str,
    ) -> None:
        """运行反思者并应用经验库操作。"""
        from trading.hypothesis import build_hypothesis_check_context

        # 反思执行结果，失败/降级不阻断主流程
        try:
            account_positions = {p.symbol: p for p in account.positions}
            prev_ctx = build_hypothesis_check_context(self.hypotheses, account_positions)
            decision_summary = json.dumps({
                "market_assessment": decision.market_assessment,
                "instructions": [
                    {"symbol": i.symbol, "action": i.action.value,
                     "quantity": i.quantity, "leverage": i.leverage,
                     "stop_loss": i.stop_loss, "take_profit": i.take_profit,
                     "reason": i.reason}
                    for i in decision.instructions
                ],
            }, ensure_ascii=False, indent=2)
            reflection = self.reflector.reflect(
                market_report=market_report,
                decision_summary=decision_summary,
                execution_results=exec_results,
                account_context=_format_account_ctx(account),
                experience_context=self.experiences.format_for_context(),
                previous_context=prev_ctx,
            )
            self._last_reflection = {
                "self_assessment": reflection.self_assessment,
                "severe_loss": reflection.severe_loss,
            }
            logger.info("反思评估: %s", reflection.self_assessment[:120])
            for op in reflection.experience_ops:
                self._apply_experience_op(op)
        except Exception as exc:  # noqa: BLE001
            logger.warning("反思环节失败（不影响本轮交易）: %s", exc)

    def _apply_experience_op(self, op) -> None:
        """应用单条经验库操作。"""
        action = op.action
        try:
            if action == ExperienceAction.WRITE:
                if not op.title or not op.content:
                    logger.warning("WRITE 操作缺少标题或内容，已忽略")
                    return
                eid = self.experiences.add(
                    category=op.category or "市场观察",
                    title=op.title,
                    content=op.content,
                )
                logger.info("经验库 WRITE #%s", eid)
            elif action == ExperienceAction.UPDATE:
                ok = self.experiences.update(op.experience_id, **{
                    "category": op.category,
                    "title": op.title,
                    "content": op.content,
                })
                if not ok:
                    logger.warning("经验库 UPDATE 失败: id=%s 不存在", op.experience_id)
            elif action == ExperienceAction.DELETE:
                ok = self.experiences.delete(op.experience_id)
                if not ok:
                    logger.warning("经验库 DELETE 失败: id=%s 不存在", op.experience_id)
            elif action == ExperienceAction.NONE:
                pass
            else:
                logger.warning("未知经验库操作: %s", action)
        except Exception as exc:  # noqa: BLE001
            logger.error("应用经验库操作失败 %s: %s", action, exc)

    def _record_hypotheses(
        self,
        passed: list,
        exec_results: list[dict],
        account,
        decision,
        price_map: dict[str, float],
    ) -> None:
        """记录开仓/调仓假设；清仓/平仓的币种清除假设。"""
        # 先处理：已被平仓/清仓的币种 → 清除假设
        closed_symbols = {
            r["symbol"]
            for r in exec_results
            if r.get("status") in ("CLOSED", "DRY_RUN")
            and r.get("action") in ("CLOSE_LONG", "CLOSE_SHORT", "FLATTEN")
        }
        # DRY_RUN 下平仓不算真实清除；真实平仓才清。DRY_RUN 保守不清除。
        if not config.dry_run:
            for sym in closed_symbols:
                self.hypotheses.remove(sym)

        # 记录开仓/调仓
        instruction_map = {i.symbol: i for i in passed}
        for r in exec_results:
            symbol = r.get("symbol")
            ins = instruction_map.get(symbol)
            if ins is None:
                continue
            action = r.get("action")
            if action in ("OPEN_LONG", "OPEN_SHORT"):
                pos_side = "LONG" if action == "OPEN_LONG" else "SHORT"
                entry = ins.price or price_map.get(symbol, 0)
                self.hypotheses.set(symbol, {
                    "side": pos_side,
                    "entry_price": entry,
                    "quantity": ins.quantity,
                    "leverage": ins.leverage,
                    "stop_loss": ins.stop_loss,
                    "take_profit": ins.take_profit,
                    "opened_at": datetime.now().isoformat(),
                    "rationale": ins.reason,
                    "assumptions": ins.reason,
                })
                logger.info("已记录 %s 开仓假设: %s", symbol, ins.reason[:100])

    def run_loop(self) -> None:
        """循环模式：每 interval 分钟执行一轮。"""
        logger.info(
            "启动循环模式，间隔 %d 分钟（测试网=%s DRY_RUN=%s）",
            self.interval_minutes, config.binance_testnet, config.dry_run,
        )
        while True:
            started = time.time()
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception("循环中出现未处理异常: %s", exc)
            elapsed = time.time() - started
            sleep_sec = max(self.interval_minutes * 60 - elapsed, 10)
            logger.info("下一轮将在 %.0f 秒后进行", sleep_sec)
            time.sleep(sleep_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="币安永续合约 AI 交易系统")
    parser.add_argument("--once", action="store_true", help="只执行一轮")
    parser.add_argument("--interval", type=int, default=None, help="循环间隔(分钟)")
    args = parser.parse_args()

    try:
        config.validate()
    except Exception as exc:  # noqa: BLE001
        logger.error("配置校验失败: %s", exc)
        raise SystemExit(1)

    system = TradingSystem(interval_minutes=args.interval)
    if args.once:
        result = system.run_once()
        # 简要输出决策摘要
        print("\n===== 决策摘要 =====")
        print("市场评估:", result.get("market_assessment", "N/A")[:300])
        for ins in result.get("instructions_after_risk", []):
            print(
                f"- {ins['symbol']} {ins['action']} "
                f"type={ins['order_type']} qty={ins['quantity']} "
                f"price={ins['price']} lev={ins['leverage']} "
                f"sl={ins['stop_loss']} tp={ins['take_profit']}"
            )
        if result.get("risk_blocked"):
            print("\n被风控拦截的指令:")
            for b in result["risk_blocked"]:
                print(f"- {b['symbol']}: {'; '.join(b['errors'])}")
        if result.get("reflection"):
            print("\n===== 自我反省 =====")
            refl = result["reflection"]
            print(f"严重亏损: {refl.get('severe_loss')}")
            print(f"评估: {refl.get('self_assessment', '')[:300]}")
        print(f"\n经验库条目数: {system.experiences.count()}（state/experience_library.json）")
    else:
        system.run_loop()


if __name__ == "__main__":
    main()
