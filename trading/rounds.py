"""轮次历史记录（RoundLog）。

每轮交易结束后将 result 精简后追加保存，供 HTTP 页面展示历史交易情况。
为节省存储：
- 只保留关键字段（账户/持仓/挂单/指令/执行结果/决策评估/风控拦截等），
  丢弃 market_report、thesis_context、news 等大文本；
- 紧凑 JSON（无缩进/空格）；
- 只保留最近 ROUNDS_KEEP 轮，避免文件无限增长。

存储位置：state/rounds.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 只保留最近多少轮
ROUNDS_KEEP = 50


class RoundLog:
    """轮次历史记录的持久化存储（追加 + 截断 + 紧凑 JSON）。"""

    def __init__(self, path: Optional[Path] = None, keep: int = ROUNDS_KEEP) -> None:
        self.path = path or (Path(__file__).resolve().parent.parent / "state" / "rounds.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self._data: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("轮次历史文件损坏，重新初始化: %s", exc)
            return []

    def save(self) -> None:
        # 紧凑 JSON（无缩进/空格），占用存储较少
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, separators=(",", ":"))

    def append(self, result: dict[str, Any]) -> None:
        """追加一轮精简记录，并截断到最近 keep 轮。"""
        summary = self._summarize(result)
        self._data.append(summary)
        if len(self._data) > self.keep:
            self._data = self._data[-self.keep:]
        self.save()

    def all(self) -> list[dict]:
        return list(self._data)

    def count(self) -> int:
        return len(self._data)

    @staticmethod
    def _summarize(result: dict[str, Any]) -> dict[str, Any]:
        """从完整 result 中抽取关键字段，丢弃大文本以节省存储。"""
        keep_keys = (
            "timestamp", "symbols", "testnet", "dry_run",
            "account", "open_orders", "min_margin_context",
            "market_assessment", "risk_notes",
            "thesis_ops", "watch_conditions", "risk_blocked",
            "instructions_after_risk", "confirmations", "execution",
            "thesis_count", "reflection", "error",
        )
        return {k: result.get(k) for k in keep_keys if k in result}
