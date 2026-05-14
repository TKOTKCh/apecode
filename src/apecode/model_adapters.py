"""
模型适配器层 - 用于 nano agent，使用官方提供商 SDK

核心功能：
- 提供统一接口，支持多个模型提供商（OpenAI、Anthropic、Kimi）
- 在 OpenAI 和 Anthropic 格式之间进行消息和工具调用转换
- 统一的错误处理
- 延迟加载 SDK 依赖

支持的提供商：
- OpenAI（兼容 Kimi 等 OpenAI 格式的 API）
- Anthropic
- Kimi（继承自 OpenAI 适配器）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


class ModelError(RuntimeError):
    """模型调用失败时抛出的异常"""


def _coerce_text_content(content: Any) -> str:
    """
    将任意内容强制转换为文本字符串
    
    支持的内容类型：
    - None → 空字符串
    - str → 直接返回
    - list → 提取其中 type="text" 的字典项
    - 其他 → 转换为 str
    
    参数：
    - content: 任意内容
    
    返回：
    - 文本字符串
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _openai_messages_to_anthropic(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """
    将 OpenAI 格式的消息列表转换为 Anthropic 格式
    
    转换逻辑：
    - system 角色 → 提取到 system_prompt
    - user 角色 → 转换为 Anthropic user 消息
    - assistant 角色 → 转换为 Anthropic assistant 消息（包含文本和 tool_use）
    - tool 角色 → 转换为 Anthropic tool_result（包装在 user 消息中）
    
    参数：
    - messages: OpenAI 格式的消息列表
    
    返回：
    - (system_prompt, anthropic_messages): 系统提示词和转换后的消息列表
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        # 处理 system 消息
        if role == "system":
            text = _coerce_text_content(message.get("content"))
            if text:
                system_parts.append(text)
            continue

        # 处理 user 消息
        if role == "user":
            converted.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": _coerce_text_content(message.get("content"))}],
                }
            )
            continue

        # 处理 assistant 消息（包含文本和 tool_calls）
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = _coerce_text_content(message.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            # 转换 tool_calls 为 tool_use 块
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                raw_arguments = function.get("arguments", "{}")
                # 解析参数 JSON（处理解析失败的情况）
                try:
                    tool_input = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    tool_input = {"_raw_arguments": str(raw_arguments)}
                if not isinstance(tool_input, dict):
                    tool_input = {"value": tool_input}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tool_call.get("id", "")),
                        "name": str(function.get("name", "")),
                        "input": tool_input,
                    }
                )
            # 确保至少有一个块
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            converted.append({"role": "assistant", "content": blocks})
            continue

        # 处理 tool 结果消息
        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": str(message.get("tool_call_id", "")),
                            "content": _coerce_text_content(message.get("content")),
                        }
                    ],
                }
            )

    return "\n\n".join(system_parts), converted


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    将 OpenAI 格式的工具列表转换为 Anthropic 格式
    
    主要转换：
    - 提取 function.name → name
    - 提取 function.description → description
    - 提取 function.parameters → input_schema
    
    参数：
    - tools: OpenAI 格式的工具列表
    
    返回：
    - Anthropic 格式的工具列表
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") or {}
        converted.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted


def _anthropic_message_to_openai(message: dict[str, Any]) -> dict[str, Any]:
    """
    将 Anthropic 格式的消息转换为 OpenAI 格式
    
    转换逻辑：
    - text 块 → content 文本
    - tool_use 块 → tool_calls 数组
    
    参数：
    - message: Anthropic 格式的消息字典
    
    返回：
    - OpenAI 格式的消息字典
    """
    content = message.get("content") or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
            continue
        if block.get("type") == "tool_use":
            # 将 tool_use 转换为 OpenAI 的 tool_calls 格式
            tool_calls.append(
                {
                    "id": str(block.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                }
            )
    result: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _openai_message_to_dict(message: Any) -> dict[str, Any]:
    """
    将 OpenAI SDK 返回的消息对象转换为字典
    
    功能：
    - 提取 content 字段
    - 保留 reasoning_content（用于思考模型如 Kimi K2.5）
    - 转换 tool_calls 为标准字典格式
    
    参数：
    - message: OpenAI SDK 的消息对象
    
    返回：
    - 标准化的消息字典
    """
    content = message.content if hasattr(message, "content") else ""
    result: dict[str, Any] = {"role": "assistant", "content": content or ""}
    # 保留思考内容（用于 Kimi K2.5 等思考模型）
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        result["reasoning_content"] = reasoning_content
    raw_tool_calls = list(getattr(message, "tool_calls", None) or [])
    if raw_tool_calls:
        normalized_calls: list[dict[str, Any]] = []
        for item in raw_tool_calls:
            function = getattr(item, "function", None)
            normalized_calls.append(
                {
                    "id": str(getattr(item, "id", "")),
                    "type": "function",
                    "function": {
                        "name": str(getattr(function, "name", "")),
                        "arguments": str(getattr(function, "arguments", "{}")),
                    },
                }
            )
        result["tool_calls"] = normalized_calls
    return result


def _require_openai_sdk():
    """
    检查并导入 OpenAI SDK（延迟加载）
    
    如果 SDK 未安装，抛出 ModelError 并提示安装方法
    
    返回：
    - (OpenAI, APIError, APIConnectionError, APITimeoutError): SDK 类和异常类
    """
    try:
        from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
    except Exception as exc:  # pragma: no cover - depends on env
        raise ModelError("OpenAI SDK is required. Install dependency `openai` (e.g. `uv pip install openai`).") from exc
    return OpenAI, APIError, APIConnectionError, APITimeoutError


def _require_anthropic_sdk():
    """
    检查并导入 Anthropic SDK（延迟加载）
    
    如果 SDK 未安装，抛出 ModelError 并提示安装方法
    
    返回：
    - (Anthropic, APIError, APIConnectionError, APITimeoutError): SDK 类和异常类
    """
    try:
        from anthropic import Anthropic, APIConnectionError, APIError, APITimeoutError
    except Exception as exc:  # pragma: no cover - depends on env
        raise ModelError("Anthropic SDK is required. Install dependency `anthropic` (e.g. `uv pip install anthropic`).") from exc
    return Anthropic, APIError, APIConnectionError, APITimeoutError


@dataclass(slots=True)
class OpenAIChatCompletionsClient:
    """
    基于官方 OpenAI SDK 的兼容聊天客户端
    
    支持所有 OpenAI 格式的 API（如 Kimi、DeepSeek 等）
    
    字段：
    - api_key: API 密钥
    - model: 模型名称
    - base_url: API 基础 URL
    - timeout: 请求超时时间（秒）
    - temperature: 温度参数
    - _client: 内部 SDK 客户端（不初始化，不显示在 repr）
    """

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 120
    temperature: float = 0.0
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """初始化后创建 OpenAI 客户端"""
        OpenAI, _api_error, _conn_error, _timeout_error = _require_openai_sdk()
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        发送一次补全请求并返回一条 assistant 消息
        
        参数：
        - messages: 消息历史列表（OpenAI 格式）
        - tools: 工具定义列表（OpenAI 格式）
        
        返回：
        - 标准化的 assistant 消息字典
        
        异常：
        - ModelError: 超时、网络错误、提供商错误等
        """
        _OpenAI, APIError, APIConnectionError, APITimeoutError = _require_openai_sdk()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.temperature,
            )
        except APITimeoutError as exc:
            raise ModelError("Request timed out") from exc
        except APIConnectionError as exc:
            raise ModelError(f"Network error: {exc}") from exc
        except APIError as exc:
            raise ModelError(f"Provider error: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelError(f"Unexpected OpenAI SDK error: {exc}") from exc

        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, TypeError) as exc:
            raise ModelError(f"Unexpected model response: {response}") from exc
        return _openai_message_to_dict(message)


@dataclass(slots=True)
class AnthropicMessagesClient:
    """
    Anthropic Messages API 适配器，提供类 OpenAI 的工具调用格式
    
    内部会进行 OpenAI ↔ Anthropic 格式转换
    
    字段：
    - api_key: API 密钥
    - model: 模型名称
    - base_url: API 基础 URL
    - api_version: API 版本
    - timeout: 请求超时时间（秒）
    - max_tokens: 最大生成 token 数
    - temperature: 温度参数
    - _client: 内部 SDK 客户端（不初始化，不显示在 repr）
    """

    api_key: str
    model: str
    base_url: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"
    timeout: int = 120
    max_tokens: int = 4096
    temperature: float = 0.0
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """初始化后创建 Anthropic 客户端"""
        Anthropic, _api_error, _conn_error, _timeout_error = _require_anthropic_sdk()
        self._client = Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            default_headers={"anthropic-version": self.api_version},
        )

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        发送一次补全请求并返回一条 assistant 消息（OpenAI 格式）
        
        内部流程：
        1. 将 OpenAI 消息转换为 Anthropic 格式
        2. 调用 Anthropic API
        3. 将 Anthropic 响应转换回 OpenAI 格式
        
        参数：
        - messages: 消息历史列表（OpenAI 格式）
        - tools: 工具定义列表（OpenAI 格式）
        
        返回：
        - 标准化的 assistant 消息字典（OpenAI 格式）
        
        异常：
        - ModelError: 超时、网络错误、提供商错误等
        """
        system_prompt, anthropic_messages = _openai_messages_to_anthropic(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tools": _openai_tools_to_anthropic(tools),
        }
        if system_prompt:
            payload["system"] = system_prompt

        _Anthropic, APIError, APIConnectionError, APITimeoutError = _require_anthropic_sdk()
        try:
            response = self._client.messages.create(**payload)
        except APITimeoutError as exc:
            raise ModelError("Request timed out") from exc
        except APIConnectionError as exc:
            raise ModelError(f"Network error: {exc}") from exc
        except APIError as exc:
            raise ModelError(f"Provider error: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelError(f"Unexpected Anthropic SDK error: {exc}") from exc

        body = response.model_dump(mode="json", exclude_none=True)
        if not isinstance(body, dict):
            raise ModelError(f"Unexpected model response: {response}")
        return _anthropic_message_to_openai(body)


@dataclass(slots=True)
class KimiChatCompletionsClient(OpenAIChatCompletionsClient):
    """Kimi OpenAI 兼容适配器（继承自 OpenAI 适配器）"""


def create_model_client(*, provider: str, model: str, timeout: int, temperature: float | None = None):
    """
    根据提供商名称创建模型客户端
    
    支持的提供商：
    - openai: 使用 OPENAI_API_KEY 和 OPENAI_BASE_URL
    - anthropic: 使用 ANTHROPIC_API_KEY、ANTHROPIC_BASE_URL、ANTHROPIC_API_VERSION
    - kimi: 使用 KIMI_API_KEY 和 KIMI_BASE_URL（继承自 OpenAI 适配器）
    
    参数：
    - provider: 提供商名称（不区分大小写）
    - model: 模型名称
    - timeout: 超时时间（秒）
    - temperature: 温度参数（可选，None 则使用默认值）
    
    返回：
    - 对应的模型客户端实例
    
    异常：
    - RuntimeError: API key 缺失或不支持的提供商
    """
    normalized = provider.strip().lower()
    if normalized == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return OpenAIChatCompletionsClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature if temperature is not None else 0.0,
        )

    if normalized == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for provider=anthropic")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        api_version = os.environ.get("ANTHROPIC_API_VERSION", "2023-06-01")
        return AnthropicMessagesClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            api_version=api_version,
            timeout=timeout,
            temperature=temperature if temperature is not None else 0.0,
        )

    if normalized == "kimi":
        api_key = os.environ.get("KIMI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("KIMI_API_KEY is required for provider=kimi")
        base_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
        return KimiChatCompletionsClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature if temperature is not None else 1.0,
        )

    raise RuntimeError(f"unsupported provider: {provider}")
