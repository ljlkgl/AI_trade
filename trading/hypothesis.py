"""持仓假设存储。

记录每轮开仓（调仓）时的完整理由与假设，供下一轮决策时检查
当前行情是否偏离了原本的假设（假设验证机制）。

存储位置：state/position_hypotheses.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HypothesisStore:
    """持仓假设的持久化存储。"""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (Path(__file__).resolve().parent.parent / "state" / "position_hypotheses.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("假设状态文件损坏，重新初始化: %s", exc)
            return {}

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get(self, symbol: str) -> Optional[dict]:
        return self._data.get(symbol)

    def set(self, symbol: str, record: dict) -> None:
        """写入一条假设记录（开仓/调仓时）。"""
        self._data[symbol] = record
        self.save()
        logger.info("已记录 %s 的持仓假设", symbol)

    def remove(self, symbol: str) -> None:
        """清除假设（清仓/止损平仓时）。"""
        if symbol in self._data:
            del self._data[symbol]
            self.save()
            logger.info("已清除 %s 的持仓假设", symbol)

    def all(self) -> dict[str, dict]:
        return dict(self._data)

    def symbols(self) -> list[str]:
        return list(self._data.keys())


def build_hypothesis_check_context(
    store: HypothesisStore, account_positions: dict[str, Any]
) -> str:
    """构建「原始假设 vs 当前行情」偏离检查上下文。

    对每个有假设记录的币种：
    - 若账户已无该币种持仓：说明假设已失效（已平仓），无需检查
    - 若仍有持仓：注入原始假设与当前持仓/价格，供模型判断行情是否偏离
    """
    lines = ["# 持仓假设检查（上次开仓/调仓理由 vs 当前行情）"]
    has_content = False
    for symbol, record in store.all().items():
        pos = account_positions.get(symbol)
        if pos is None:
            lines.append(
                f"- {symbol}: 存在历史假设但当前无持仓（可能已平仓/止损），"
                f"本轮按从零分析处理"
            )
            continue
        has_content = True
        lines.append(f"## {symbol}")
        lines.append(f"- 开仓时间: {record.get('opened_at', 'N/A')}")
        lines.append(f"- 方向: {record.get('side', 'N/A')}  开仓均价: {record.get('entry_price', 'N/A')}")
        lines.append(f"- 杠杆: {record.get('leverage', 'N/A')}x  止损: {record.get('stop_loss', 'N/A')}  止盈: {record.get('take_profit', 'N/A')}")
        lines.append(f"- 原始开仓理由: {record.get('rationale', 'N/A')}")
        lines.append(f"- 原始假设: {record.get('assumptions', 'N/A')}")
        lines.append(
            f"- 当前持仓: 数量 {abs(pos.position_amt):.6g}  标记价 {pos.mark_price:.6g}  "
            f"未实现盈亏 {pos.unrealized_pnl:+.4f}"
        )
        # 偏离程度提示
        entry = record.get("entry_price")
        if isinstance(entry, (int, float)) and entry > 0:
            drift = (pos.mark_price / entry - 1) * 100
            lines.append(
                f"- 相对开仓价偏离: {drift:+.2f}%（{'顺向' if (drift > 0) == (record.get('side') == 'LONG') else '逆向'}）"
            )
        lines.append("")

    if not has_content:
        lines.append("（无现存持仓假设记录）")
    return "\n".join(lines)
