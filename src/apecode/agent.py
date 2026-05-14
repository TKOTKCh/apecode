"""
Nano 代码代理的核心循环模块

核心功能：
- 提供聊天模型协议接口（ChatModel）
- 定义代理配置（AgentConfig）
- 提供回调钩子（AgentCallbacks）
- 实现工具调用循环（NanoCodeAgent）

工作流程：
1. 用户输入 → 添加到消息历史
2. 模型完成 → 选择工具或直接回答
3. 如果有工具调用 → 执行工具 → 添加结果到消息历史
4. 循环直到模型不再调用工具或达到最大步数
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from apecode.tools import ToolRegistry


class ChatModel(Protocol):
    """
    聊天模型适配器协议
    
    任何实现这个协议的类都可以作为 NanoCodeAgent 的模型使用
    """

    def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """
        执行一次聊天补全
        
        参数：
        - messages: 消息历史列表（OpenAI 格式）
        - tools: 工具定义列表（OpenAI 格式）
        
        返回：
        - 助手消息字典
        """


def _coerce_text(content: Any) -> str:
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


@dataclass(slots=True)
class AgentConfig:
    """
    代理执行的运行时配置
    
    字段：
    - max_steps: 最大执行步数（防止无限循环）
    """

    max_steps: int = 20


@dataclass(slots=True)
class AgentCallbacks:
    """
    可选的事件回调
    
    保持代理框架无关 — 不导入 Rich 等特定库
    
    字段：
    - on_status: 状态变化回调（"Thinking..." 或空字符串清除）
    - on_thinking: 思考内容回调（模型返回 reasoning_content 时）
    - on_tool_call: 工具调用前回调（tool_name, arguments_json）
    - on_tool_result: 工具调用后回调（tool_name, result_text）
    """

    on_status: Callable[[str], None] | None = None
    """Called with status text ("Thinking...") or empty string to clear."""

    on_thinking: Callable[[str], None] | None = None
    """Called when the model returns reasoning_content."""

    on_tool_call: Callable[[str, str], None] | None = None
    """Called before tool execution with (tool_name, arguments_json)."""

    on_tool_result: Callable[[str, str], None] | None = None
    """Called after tool execution with (tool_name, result_text)."""


class NanoCodeAgent:
    """
    一个小巧的工具调用循环，使用 Chat Completions API
    
    核心特性：
    - 保持对话历史
    - 支持工具调用
    - 支持回调钩子
    - 可配置最大步数
    """

    def __init__(
        self,
        *,
        model: ChatModel,
        tools: ToolRegistry,
        system_prompt: str,
        config: AgentConfig | None = None,
        callbacks: AgentCallbacks | None = None,
        # Legacy single callback kept for backwards compat with tests
        on_tool_call: Callable[[str, str], None] | None = None,
    ) -> None:
        """
        初始化 NanoCodeAgent
        
        参数：
        - model: 聊天模型（实现 ChatModel 协议）
        - tools: 工具注册表
        - system_prompt: 系统提示词
        - config: 代理配置（可选）
        - callbacks: 回调钩子（可选）
        - on_tool_call: 旧版单回调（保持与测试的兼容性）
        """
        self.model = model
        self.tools = tools
        self.config = config or AgentConfig()
        self.cb = callbacks or AgentCallbacks(on_tool_call=on_tool_call)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def _fire(self, name: str, *args: Any) -> None:
        """
        触发回调事件（内部方法）
        
        参数：
        - name: 回调方法名
        - args: 回调参数
        """
        fn = getattr(self.cb, name, None)
        if fn is not None:
            fn(*args)

    def run(self, user_input: str) -> str:
        """
        运行一个用户回合直到完成
        
        执行流程：
        1. 添加用户消息到历史
        2. 循环执行直到没有工具调用或达到最大步数：
           a. 调用模型完成
           b. 如果有工具调用 → 执行工具 → 添加结果到历史
           c. 如果没有工具调用 → 返回助手内容
        
        参数：
        - user_input: 用户输入文本
        
        返回：
        - 最终助手回复文本
        
        异常：
        - RuntimeError: 超过最大步数
        """
        self.messages.append({"role": "user", "content": user_input})
        for _ in range(self.config.max_steps):
            # 显示"思考中"状态
            self._fire("on_status", "Thinking...")
            # 调用模型生成回复
            assistant = self.model.complete(
                messages=self.messages,
                tools=self.tools.as_openai_tools(),
            )
            # 清除状态
            self._fire("on_status", "")

            # 如果模型有思考内容（如 Claude 3 的 reasoning_content），显示出来
            reasoning = assistant.get("reasoning_content")
            if reasoning:
                self._fire("on_thinking", str(reasoning))

            # 处理模型回复
            tool_calls = assistant.get("tool_calls") or []
            assistant_record: dict[str, Any] = {
                "role": "assistant",
                "content": assistant.get("content"),
            }
            # 保存模型特定字段（例如思考模型的 reasoning_content）
            for key in ("reasoning_content",):
                if assistant.get(key):
                    assistant_record[key] = assistant[key]
            if tool_calls:
                assistant_record["tool_calls"] = tool_calls
            self.messages.append(assistant_record)

            # 如果没有工具调用，说明完成了，返回回复
            if not tool_calls:
                return _coerce_text(assistant.get("content"))

            # 执行所有工具调用
            for call in tool_calls:
                call_id = str(call.get("id", ""))
                function = call.get("function") or {}
                name = str(function.get("name", ""))
                arguments = str(function.get("arguments", "{}"))
                # 触发工具调用前回调
                self._fire("on_tool_call", name, arguments)
                # 执行工具
                result = self.tools.execute(name, arguments)
                # 触发工具调用后回调
                self._fire("on_tool_result", name, result)
                # 添加工具结果到消息历史
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result,
                    }
                )

        # 超过最大步数，抛出异常
        raise RuntimeError(f"max steps exceeded ({self.config.max_steps})")
