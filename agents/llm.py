"""LLM 客户端：OpenAI 兼容接口，支持结构化 JSON 输出。

复用 TradingAgents 的思路：模型输出必须遵循指定 JSON 格式，
这里通过 response_format=json_object + 强约束 prompt 实现，
并对输出做 pydantic 校验与重试。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """LLM 调用或解析失败。"""


class LLMClient:
    """轻量 OpenAI 兼容客户端。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 2,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or config.llm_api_key
        self.base_url = base_url or config.llm_base_url or None
        self.model = model or config.llm_model
        self.temperature = temperature
        self.max_retries = max_retries
        # 推理强度三档（仿照 TradingAgents reasoning_effort）：low / medium / high，空=不设置
        effort = (reasoning_effort or config.llm_reasoning_effort).strip().lower()
        self.reasoning_effort = effort if effort in ("low", "medium", "high") else ""
        if not self.api_key:
            raise LLMError("LLM_API_KEY 未配置")
        if not self.model:
            raise LLMError("LLM_MODEL 未配置")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        json_mode: bool = False,
        stream: bool = True,
        label: str = "",
    ) -> str:
        """发送对话，返回文本。

        stream=True 时实时打印模型输出（便于观察模型回复/决策过程）。
        json_mode=True 时要求模型输出合法 JSON。
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if label:
            print(f"\n===== {label} =====", flush=True)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                content = self._create_completion(kwargs, stream=stream, json_mode=json_mode)
                return content
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if "reasoning_effort" in kwargs:
                    # 部分兼容接口（如 SenseNova/部分 DeepSeek 网关）不支持该参数，
                    # 移除后降级重试一次，避免整轮失败
                    logger.warning(
                        "接口不支持 reasoning_effort 参数，已降级重试: %s", exc
                    )
                    kwargs.pop("reasoning_effort")
                    continue
                logger.warning("LLM 调用失败（第 %d 次）: %s", attempt + 1, exc)
        raise LLMError(f"LLM 调用重试后仍失败: {last_exc}")

    def _create_completion(
        self, kwargs: dict, stream: bool, json_mode: bool
    ) -> str:
        """执行一次 completion；流式时逐块打印。"""
        if stream:
            try:
                stream_resp = self.client.chat.completions.create(
                    **kwargs, stream=True
                )
            except Exception as exc:  # noqa: BLE001
                # 部分兼容接口不支持流式，回退到非流式
                logger.warning("流式请求失败，回退非流式: %s", exc)
                return self._create_completion(kwargs, stream=False, json_mode=json_mode)

            pieces: list[str] = []
            for chunk in stream_resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None) or ""
                if token:
                    pieces.append(token)
                    print(token, end="", flush=True)
            print(flush=True)  # 结束换行
            content = "".join(pieces)
            if json_mode:
                content = _extract_json(content)
                json.loads(content)  # 校验合法 JSON
            return content

        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        if json_mode:
            content = _extract_json(content)
            json.loads(content)
        return content

    def chat_json(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        label: str = "",
    ) -> dict:
        """返回解析后的 dict。"""
        content = self.chat(
            messages, temperature=temperature, json_mode=True, label=label
        )
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"模型输出非法 JSON: {exc}\n原文: {content[:500]}") from exc
        if not isinstance(data, dict):
            raise LLMError("模型输出 JSON 不是对象")
        return data


def _extract_json(text: str) -> str:
    """从模型文本中提取 JSON 对象（处理代码块包裹等情况）。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text
