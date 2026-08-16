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

from agents.confirmer import Confirmer
from agents.decision_maker import DecisionMaker, format_account_context
from agents.llm import AllLLMUnavailable, LLMClient
from agents.market_analyst import MarketAnalyst
from agents.reflector import Reflector
from agents.schemas import ConfirmationAction, ExperienceAction
from config import config
from trading.binance_client import BinanceClient
from trading.experience import ExperienceStore
from trading.executor import OrderExecutor
from trading.hypothesis import HypothesisStore, build_hypothesis_check_context
from trading.market import MarketDataService
from trading.news import NewsService
from trading.risk import RiskManager
from trading.watch import WatchStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trading_system")


def _format_account_ctx(account, open_orders_by_symbol=None) -> str:
    """账户现状 markdown（余额/持仓/未成交挂单），供确认者与反思者使用。"""
    return format_account_context(account, open_orders_by_symbol)


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
        # 两者都挂同一个备用 LLM（LLM_BACKUP_*）：主 API 失效时自动切换
        backup_llm = None
        if config.llm_backup_model:
            backup_llm = LLMClient(
                api_key=config.llm_backup_api_key or config.llm_api_key,
                base_url=config.llm_backup_base_url or config.llm_base_url,
                model=config.llm_backup_model,
            )
            logger.info("已启用备用 LLM: %s", backup_llm.model)
        self.llm = LLMClient(fallback=backup_llm)
        self.llm_deep = (
            LLMClient(model=config.llm_deep_model, fallback=backup_llm)
            if config.llm_deep_model
            else self.llm
        )
        self.market_data = MarketDataService(self.client)
        self.news_service = NewsService()
        self.market_analyst = MarketAnalyst(self.llm)
        self.decision_maker = DecisionMaker(self.llm_deep)
        self.reflector = Reflector(self.llm)
        # 执行前逐条确认者（用 quick 模型：单条指令的轻量复核）
        self.confirmer = Confirmer(self.llm)
        self.risk = RiskManager()
        self.executor = OrderExecutor(self.client, self.risk)
        self.hypotheses = HypothesisStore()
        self.experiences = ExperienceStore()
        self.watch_store = WatchStore(max_age_hours=config.watch_max_age_hours)
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

        # 1.5 未成交挂单（决策者/确认者管理挂单、调止盈止损必需：含订单ID）
        open_orders_by_symbol: dict[str, list] = {}
        for sym in self.symbols:
            try:
                open_orders_by_symbol[sym] = self.client.get_open_orders(sym)
            except Exception as exc:  # noqa: BLE001
                logger.warning("获取 %s 未成交挂单失败: %s", sym, exc)
                open_orders_by_symbol[sym] = []
        result["open_orders"] = {
            sym: [
                {"orderId": o.get("orderId"), "side": o.get("side"), "type": o.get("type"),
                 "price": o.get("price"), "stopPrice": o.get("stopPrice"),
                 "qty": o.get("origQty"), "filled": o.get("executedQty"),
                 "reduceOnly": o.get("reduceOnly"), "status": o.get("status")}
                for o in orders
            ]
            for sym, orders in open_orders_by_symbol.items()
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
        except AllLLMUnavailable as exc:
            return self._handle_llm_outage(exc, result)
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
                open_orders_by_symbol=open_orders_by_symbol,
            )
            logger.info(
                "决策: %d 条指令, 评估=%s",
                len(decision.instructions),
                decision.market_assessment[:120],
            )
        except AllLLMUnavailable as exc:
            return self._handle_llm_outage(exc, result)
        except Exception as exc:  # noqa: BLE001
            logger.error("决策失败: %s", exc)
            result["error"] = f"决策失败: {exc}"
            return result
        result["market_assessment"] = decision.market_assessment
        result["risk_notes"] = decision.risk_notes
        # 5.5 更新唤醒条件（模型可在正常循环外设定价格触发，全量替换）
        result["watch_conditions"] = [
            c.model_dump() for c in decision.watch_conditions
        ]
        try:
            self.watch_store.replace(decision.watch_conditions)
        except Exception as exc:  # noqa: BLE001
            logger.warning("更新唤醒条件失败: %s", exc)

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

        # 7. 逐条确认后执行：每条指令执行前调用确认者复核，
        #    确认过程中模型可 PROCEED / SKIP / REPLACE（输出修正后的新指令）
        exec_results: list[dict] = []
        executed_instructions: list = []  # 实际执行的指令（含 REPLACE 修正后）
        confirmations: list[dict] = []
        if passed:
            account_snapshot = account
            for ins in passed:
                # 刷新最新价格（前几条执行可能已影响行情判断）
                try:
                    mark = self.client.get_ticker_price(ins.symbol)
                except Exception:  # noqa: BLE001
                    mark = price_map.get(ins.symbol, 0)
                # 刷新账户（真实模式下前几条执行会改变持仓/余额）
                try:
                    account_snapshot = self.client.get_account()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("刷新账户失败（沿用轮起始快照）: %s", exc)
                # 刷新该币种未成交挂单（前几条指令可能已撤销/新挂单）
                try:
                    open_orders_by_symbol[ins.symbol] = self.client.get_open_orders(ins.symbol)
                except Exception:  # noqa: BLE001
                    pass
                prior_summary = "\n".join(
                    f"- {r.get('symbol')} {r.get('action')} -> {r.get('status')}"
                    for r in exec_results
                )
                # 执行前逐条确认
                try:
                    conf = self.confirmer.confirm(
                        ins, mark, _format_account_ctx(account_snapshot, open_orders_by_symbol),
                        prior_summary, decision.market_assessment,
                    )
                except AllLLMUnavailable as exc:
                    return self._handle_llm_outage(exc, result)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("指令确认失败，跳过 %s: %s", ins.symbol, exc)
                    exec_results.append({
                        "symbol": ins.symbol, "action": ins.action.value,
                        "status": "SKIPPED", "error": f"确认失败: {exc}",
                    })
                    continue
                confirmations.append({
                    "symbol": ins.symbol, "decision": conf.decision.value,
                    "reason": conf.reason,
                })
                if conf.decision == ConfirmationAction.SKIP:
                    logger.info("确认者 SKIP %s: %s", ins.symbol, conf.reason[:120])
                    exec_results.append({
                        "symbol": ins.symbol, "action": ins.action.value,
                        "status": "SKIPPED", "error": conf.reason,
                    })
                    continue

                final_ins = ins
                if conf.decision == ConfirmationAction.REPLACE:
                    new_ins = conf.instruction
                    info = symbol_info_map.get(new_ins.symbol)
                    m = price_map.get(new_ins.symbol, 0)
                    if info is None or m <= 0:
                        logger.warning("REPLACE 缺少 %s 价格/精度信息，跳过", new_ins.symbol)
                        exec_results.append({
                            "symbol": new_ins.symbol, "action": "REPLACE",
                            "status": "SKIPPED", "error": "缺少价格/精度信息",
                        })
                        continue
                    r = self.risk.validate_instruction(new_ins, account_snapshot, info, m)
                    if not r.ok:
                        logger.warning(
                            "REPLACE 新指令未过风控，跳过 %s: %s",
                            new_ins.symbol, "; ".join(r.errors),
                        )
                        exec_results.append({
                            "symbol": new_ins.symbol, "action": "REPLACE",
                            "status": "SKIPPED", "error": "; ".join(r.errors),
                        })
                        continue
                    final_ins = new_ins
                    logger.info("确认者 REPLACE %s -> 新指令已过风控", new_ins.symbol)

                # 执行单条
                try:
                    r = self.executor.execute([final_ins], account_snapshot)[0]
                except Exception as exc:  # noqa: BLE001
                    r = {"symbol": final_ins.symbol, "action": final_ins.action.value,
                         "status": "FAILED", "error": str(exc)}
                exec_results.append(r)
                executed_instructions.append(final_ins)
                order_id = r.get("order_id")
                logger.info(
                    "执行结果 %s %s -> %s%s%s",
                    r.get("symbol"), r.get("action"), r.get("status"),
                    " (DRY_RUN)" if config.dry_run else "",
                    f" orderId={order_id}" if order_id else "",
                )
            result["confirmations"] = confirmations
        else:
            logger.info("无通过风控的指令，本轮不交易")
        result["execution"] = exec_results

        # 8. 记录开仓/调仓理由到假设存储（供下一轮检查行情是否偏离）
        self._record_hypotheses(
            executed_instructions, exec_results, account, decision, price_map
        )

        # 9. 反思者复盘本轮，自主维护经验库（写入/修改/删除）
        self._reflect_and_apply(
            decision, exec_results, account, hypothesis_context, market_report,
            open_orders_by_symbol=open_orders_by_symbol,
        )
        if self._last_reflection:
            result["reflection"] = self._last_reflection

        logger.info("===== 本轮结束 =====")
        return result

    def _handle_llm_outage(self, exc, result: dict) -> dict:
        """所有 LLM API 均无法正常通讯：立即市价平掉全部持仓，本轮结束。

        由 LLMClient 在「主→备→按间隔确认多次仍失败」后抛出的
        AllLLMUnavailable 触发；平仓不依赖 LLM，仅用币安 API（reduceOnly）。
        """
        logger.error("所有 LLM API 均不可用，触发紧急平仓: %s", exc)
        try:
            account = self.client.get_account()
            emerg = self.executor.flatten_all(account, reason="LLM API 全部不可用")
            result["emergency_flatten"] = emerg
            for r in emerg:
                logger.warning(
                    "紧急平仓 %s %s -> %s",
                    r.get("symbol"), r.get("side"), r.get("status"),
                )
        except Exception as e:  # noqa: BLE001
            logger.error("紧急平仓失败: %s", e)
            result["emergency_flatten_error"] = str(e)
        result["error"] = f"所有 LLM API 均无法正常通讯，已执行紧急平仓: {exc}"
        return result

    def _emergency_flatten(self, reason: str) -> None:
        """仅执行紧急平仓（无 result 上下文时用，如反思环节）。"""
        try:
            account = self.client.get_account()
            self.executor.flatten_all(account, reason=reason)
        except Exception as exc:  # noqa: BLE001
            logger.error("紧急平仓失败: %s", exc)

    def _reflect_and_apply(
        self,
        decision,
        exec_results: list[dict],
        account,
        hypothesis_context: str,
        market_report: str,
        open_orders_by_symbol: dict | None = None,
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
                account_context=_format_account_ctx(account, open_orders_by_symbol),
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
        except AllLLMUnavailable as exc:
            # 反思环节 LLM 全部不可用：本轮刚执行过交易，立即紧急平仓撤退
            logger.error("反思环节所有 LLM 均不可用，执行紧急平仓: %s", exc)
            self._emergency_flatten("反思环节 LLM API 全部不可用")
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
            # 仅记录真正执行的 OPEN（DRY_RUN 演练 / OPENED 成交），失败/跳过不写假设
            if action in ("OPEN_LONG", "OPEN_SHORT") and r.get("status") in ("DRY_RUN", "OPENED"):
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
        """循环模式：每 interval 分钟执行一轮；若模型设定唤醒条件，条件满足时提前执行。"""
        logger.info(
            "启动循环模式，间隔 %d 分钟（测试网=%s DRY_RUN=%s 条件唤醒=%s）",
            self.interval_minutes, config.binance_testnet, config.dry_run,
            config.watch_enabled,
        )
        while True:
            started = time.time()
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception("循环中出现未处理异常: %s", exc)
            elapsed = time.time() - started
            sleep_sec = max(self.interval_minutes * 60 - elapsed, 10)
            if config.watch_enabled and self.watch_store.count() > 0:
                if self._wait_for_trigger_or_sleep(sleep_sec):
                    logger.info("唤醒条件触发，提前开始新一轮分析")
                    continue
                logger.info("等待 %d 秒无唤醒条件触发，按计划进入下一轮", sleep_sec)
            else:
                logger.info("下一轮将在 %.0f 秒后进行", sleep_sec)
                time.sleep(sleep_sec)

    def _wait_for_trigger_or_sleep(self, sleep_sec: float) -> bool:
        """等待 sleep_sec 秒，期间按 WATCH_CHECK_INTERVAL 轮询唤醒条件；满足则提前返回 True。"""
        deadline = time.time() + sleep_sec
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            wait = min(max(config.watch_check_interval, 1), remaining)
            time.sleep(wait)
            if self._check_watch_triggers():
                return True

    def _check_watch_triggers(self) -> bool:
        """轮询当前价并核对唤醒条件；任一满足则清除条件并返回 True。"""
        triggered: list[tuple[dict, float]] = []
        for t in self.watch_store.all():
            try:
                price = self.client.get_ticker_price(t["symbol"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("检查 %s 唤醒条件失败: %s", t["symbol"], exc)
                continue
            cond, val = t.get("condition"), t.get("value")
            if cond == "price_above" and price >= val:
                triggered.append((t, price))
            elif cond == "price_below" and price <= val:
                triggered.append((t, price))
        if triggered:
            detail = "; ".join(
                f"{t['symbol']} {t['condition']}@{t['value']:g}（现价 {price:.6g}）"
                for t, price in triggered
            )
            logger.warning("唤醒条件触发: %s", detail)
            self.watch_store.clear_all(reason="条件已触发")
            return True
        return False


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
