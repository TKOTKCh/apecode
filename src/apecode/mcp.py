"""
基于 fastmcp Client 的 MCP (Model Context Protocol) 集成模块

核心功能：
- 解析 .mcp.json 配置文件中的 MCP 服务器定义
- 连接 MCP 服务器并发现其提供的工具
- 将 MCP 工具桥接到 apecode 的 ToolRegistry
- 命名空间格式：mcp__服务器名__工具名
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastmcp import Client

from apecode.tools import Tool, ToolRegistry


def _sanitize_name(value: str) -> str:
    """
    清理名称，只保留字母、数字和下划线，转小写
    
    用于生成安全的标识符（如工具名、服务器名）
    """
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower() or "tool"


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """
    单个 MCP stdio 服务器的配置
    
    字段：
    - name: 服务器名称（用于命名空间）
    - command: 启动服务器的命令（如 npx、python）
    - args: 命令参数列表
    - timeout_sec: 连接和调用超时时间（秒）
    """

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    timeout_sec: int = 30


@dataclass(slots=True)
class McpBridge:
    """
    MCP 工具加载结果
    
    字段：
    - tool_names: 成功加载的工具名称列表
    - errors: 加载过程中的错误信息
    
    注意：这个精简实现每次调用工具时都会创建新会话，不需要保持长连接
    """

    tool_names: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def close(self) -> None:
        """
        关闭资源（空操作）
        
        在这个精简实现中，会话是每次请求时创建的，不需要保持长连接，
        所以 close 方法不需要做任何事情。
        """


def _parse_mcp_config(path: Path) -> list[McpServerConfig]:
    """
    解析 .mcp.json 配置文件，提取 mcpServers 条目
    
    配置文件格式示例：
    {
      "mcpServers": {
        "server-name": {
          "command": "npx",
          "args": ["-y", "mcp-server-example"],
          "timeout_sec": 30
        }
      }
    }
    
    参数：
    - path: 配置文件路径
    
    返回：
    - 解析后的 MCP 服务器配置列表
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    servers: list[McpServerConfig] = []

    raw_mcp_servers = payload.get("mcpServers")
    if isinstance(raw_mcp_servers, dict):
        for name, item in raw_mcp_servers.items():
            if not isinstance(item, dict):
                continue
            command = str(item.get("command", "")).strip()
            if not command:
                continue
            raw_args = item.get("args", [])
            args = [str(value) for value in raw_args] if isinstance(raw_args, list) else []
            # 超时时间限制在 5-300 秒之间
            timeout_sec = max(5, min(int(item.get("timeout_sec", 30)), 300))
            servers.append(
                McpServerConfig(
                    name=str(name).strip(),
                    command=command,
                    args=args,
                    timeout_sec=timeout_sec,
                )
            )

    return servers


def _make_client(server: McpServerConfig) -> Client:
    """
    为单个 MCP 服务器创建 fastmcp Client
    
    参数：
    - server: MCP 服务器配置
    
    返回：
    - 配置好的 fastmcp Client 实例
    """
    config = {"mcpServers": {server.name: {"command": server.command, "args": server.args}}}
    return Client(config)


async def _list_tools_async(server: McpServerConfig) -> list[Any]:
    """
    异步列出 MCP 服务器提供的所有工具
    
    参数：
    - server: MCP 服务器配置
    
    返回：
    - 工具描述对象列表
    """
    async with _make_client(server) as client:
        return await client.list_tools()


async def _call_tool_async(server: McpServerConfig, tool_name: str, arguments: dict[str, Any]) -> Any:
    """
    异步调用 MCP 服务器的工具
    
    参数：
    - server: MCP 服务器配置
    - tool_name: 工具名称
    - arguments: 工具参数字典
    
    返回：
    - 工具执行结果
    """
    async with _make_client(server) as client:
        return await client.call_tool(tool_name, arguments)


def _render_tool_result(result: Any, *, server_name: str, tool_name: str) -> str:
    """
    将 MCP 工具的结果渲染为字符串
    
    处理逻辑：
    1. 检查是否是错误结果
    2. 提取内容中的文本部分
    3. 处理其他类型的内容（序列化或转字符串）
    4. 拼接所有内容并返回
    
    参数：
    - result: MCP 工具返回的结果对象
    - server_name: 服务器名称（用于错误信息）
    - tool_name: 工具名称（用于错误信息）
    
    返回：
    - 渲染后的字符串
    """
    is_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
    content = list(getattr(result, "content", []) or [])

    chunks: list[str] = []
    for item in content:
        item_type = str(getattr(item, "type", ""))
        # 优先处理文本类型
        if item_type == "text":
            text = str(getattr(item, "text", ""))
            if text:
                chunks.append(text)
            continue
        # 处理有 model_dump 方法的对象（Pydantic 模型）
        if hasattr(item, "model_dump"):
            chunks.append(json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
        else:
            chunks.append(str(item))

    # 拼接所有内容片段
    rendered = "\n".join(part for part in chunks if part.strip()).strip()
    if not rendered:
        rendered = f"MCP `{server_name}/{tool_name}` returned empty result."
    if is_error:
        return f"MCP `{server_name}/{tool_name}` failed: {rendered}"
    return rendered


def load_mcp_tools(registry: ToolRegistry, config_paths: list[Path]) -> McpBridge:
    """
    从 MCP 配置文件加载工具并注册到 ToolRegistry
    
    执行流程：
    1. 遍历所有配置文件路径
    2. 解析每个配置文件中的 MCP 服务器
    3. 连接每个服务器并获取工具列表
    4. 将每个 MCP 工具包装为 apecode Tool 并注册
    5. 记录错误但继续处理其他服务器
    
    参数：
    - registry: 工具注册表，用于注册 MCP 工具
    - config_paths: MCP 配置文件路径列表
    
    返回：
    - 包含加载结果的 McpBridge 对象
    """
    bridge = McpBridge()
    seen_servers: set[str] = set()

    for raw_path in config_paths:
        path = raw_path.expanduser().resolve()
        if not path.exists() or not path.is_file():
            continue
        try:
            # 解析 MCP 配置文件，获取MCP服务器配置
            servers = _parse_mcp_config(path)
        except Exception as exc:
            bridge.errors.append(f"invalid MCP config `{path}`: {exc}")
            continue

        for server in servers:
            # 避免重复加载同名服务器
            if server.name in seen_servers:
                continue
            seen_servers.add(server.name)
            
            # 连接服务器并获取工具列表
            try:
                # 第1步：asyncio.run() 会阻塞等待，直到函数内的异步函数完成
                tools = asyncio.run(asyncio.wait_for(_list_tools_async(server), timeout=server.timeout_sec))
            except Exception as exc:
                bridge.errors.append(f"MCP server `{server.name}` unavailable: {exc}")
                continue

            # 注册每个工具
            for raw_tool in tools:
                raw_name = str(getattr(raw_tool, "name", "")).strip()
                if not raw_name:
                    continue

                # 生成带命名空间的工具名：mcp__服务器名__工具名
                namespaced = f"mcp__{_sanitize_name(server.name)}__{_sanitize_name(raw_name)}"
                description = str(getattr(raw_tool, "description", "")).strip()
                schema = getattr(raw_tool, "inputSchema", None)
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}

                # 检查是否是只读操作（影响 mutating 标记）
                annotations = getattr(raw_tool, "annotations", None)
                read_only = bool(getattr(annotations, "readOnlyHint", False)) if annotations else False

                # 构建工具处理器（使用闭包捕获 server 和 tool_name）
                def _handler(_ctx, args, *, _server=server, _tool_name=raw_name):
                    try:
                        # handler异步调用MCP服务器的工具，获取结果，这里虽然是异步调用但是asyncio会阻塞等待
                        result = asyncio.run(
                            asyncio.wait_for(
                                _call_tool_async(_server, _tool_name, args),
                                timeout=_server.timeout_sec,
                            )
                        )
                    except Exception as exc:
                        return f"MCP `{_server.name}/{_tool_name}` invocation error: {exc}"
                    return _render_tool_result(
                        result,
                        server_name=_server.name,
                        tool_name=_tool_name,
                    )

                # 注册工具
                registry.register(
                    Tool(
                        name=namespaced,
                        description=description or f"[mcp:{server.name}] call MCP tool `{raw_name}`",
                        parameters=schema,
                        handler=_handler,
                        mutating=not read_only,
                    )
                )
                bridge.tool_names.append(namespaced)

    return bridge
