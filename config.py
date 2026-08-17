"""全局配置：从环境变量 / .env 读取运行参数。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    try:
        return float(val) if val is not None and val.strip() != "" else default
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    try:
        return int(val) if val is not None and val.strip() != "" else default
    except ValueError:
        return default


class Config:
    """统一配置对象，供各模块读取。"""

    def __init__(self) -> None:
        # ---- 币安 ----
        self.binance_testnet: bool = _get_bool("BINANCE_TESTNET", False)
        if self.binance_testnet:
            self.binance_api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
            self.binance_api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
        else:
            self.binance_api_key = os.getenv("BINANCE_API_KEY", "")
            self.binance_api_secret = os.getenv("BINANCE_API_SECRET", "")

        # ---- LLM ----
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "")
        self.llm_model = os.getenv("LLM_MODEL", "")
        # 深度推理模型（仿照 TradingAgents deep_think_llm）：决策者/研究经理使用
        self.llm_deep_model = os.getenv("LLM_DEEP_MODEL", "")
        # 推理强度三档（仿照 TradingAgents reasoning_effort）：low / medium / high，空=不设置
        self.llm_reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
        self.llm_temperature = _get_float("LLM_TEMPERATURE", 0.2)
        # 备用分析 LLM（主 LLM API 失效时切换）；任一项留空则回退到主 LLM 对应配置
        self.llm_backup_api_key = os.getenv("LLM_BACKUP_API_KEY", "")
        self.llm_backup_base_url = os.getenv("LLM_BACKUP_BASE_URL", "")
        self.llm_backup_model = os.getenv("LLM_BACKUP_MODEL", "")
        # 主/备用 LLM 均失效时的连通性确认：尝试次数与每次间隔（秒）
        self.llm_emergency_attempts = _get_int("LLM_EMERGENCY_ATTEMPTS", 5)
        self.llm_emergency_interval = _get_int("LLM_EMERGENCY_INTERVAL", 60)

        # ---- 交易标的 ----
        raw_symbols = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")
        self.symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]

        # ---- 交易参数 ----
        self.max_position_ratio = _get_float("MAX_POSITION_RATIO", 0.5)
        self.max_total_position_ratio = _get_float("MAX_TOTAL_POSITION_RATIO", 0.8)
        self.max_leverage = _get_int("MAX_LEVERAGE", 20)
        self.min_notional = _get_float("MIN_NOTIONAL", 20.0)
        # 单笔开仓所需初始保证金下限（USDT）；名义价值 = 保证金 × 杠杆
        self.min_margin = _get_float("MIN_MARGIN", 2.5)
        self.dry_run: bool = _get_bool("DRY_RUN", True)

        # ---- 运行参数 ----
        self.interval_minutes = _get_int("INTERVAL_MINUTES", 60)
        self.klines_limit = _get_int("KLINES_LIMIT", 500)
        self.logs_dir = BASE_DIR / "logs"
        # 条件唤醒：模型可在正常循环外设定价格触发条件，满足时提前唤醒分析
        self.watch_enabled = _get_bool("WATCH_ENABLED", True)
        self.watch_check_interval = _get_int("WATCH_CHECK_INTERVAL", 30)  # 秒
        self.watch_max_age_hours = _get_int("WATCH_MAX_AGE_HOURS", 24)  # 小时
        # 操作理由列表中「其它类」条目的超时过期时长（小时），防止上下文无限膨胀
        self.thesis_max_age_hours = _get_int("THESIS_MAX_AGE_HOURS", 72)

    def validate(self) -> None:
        """启动前校验关键配置，缺失时给出明确错误。"""
        if not self.llm_api_key or not self.llm_model:
            raise RuntimeError(
                "LLM 未配置：请设置 LLM_API_KEY / LLM_MODEL（OpenAI 兼容接口），"
                "参考 .env.example"
            )
        if not self.binance_api_key or not self.binance_api_secret:
            raise RuntimeError(
                "币安 API Key 未配置：请设置 BINANCE_API_KEY/BINANCE_API_SECRET"
                "（或测试网 BINANCE_TESTNET_API_KEY/...）"
            )
        if not 0 < self.max_position_ratio <= 1:
            raise ValueError("MAX_POSITION_RATIO 必须在 (0,1] 之间")
        if not 0 < self.max_total_position_ratio <= 1:
            raise ValueError("MAX_TOTAL_POSITION_RATIO 必须在 (0,1] 之间")


config = Config()
