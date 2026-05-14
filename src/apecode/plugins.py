"""
声明式插件加载器模块 - 用于加载工具、斜杠命令和技能插件

核心功能：
- 查找并解析 apecode_plugin.json 清单文件
- 注册插件提供的工具、命令和技能
- 工具通过外部命令实现，支持 stdin/stdout 通信
- 命名空间隔离：插件工具名格式为 插件名__工具名
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apecode.skills import Skill
from apecode.tools import Tool, ToolContext, ToolRegistry

# 插件清单文件名，固定为 apecode_plugin.json
PLUGIN_MANIFEST_NAME = "apecode_plugin.json"


def _sanitize_name(value: str) -> str:
    """
    清理名称，只保留字母、数字和下划线，转小写
    
    用于生成安全的标识符（如工具名、命令名）
    """
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower() or "item"


def _iter_manifest_files(plugin_dirs: list[Path]) -> list[Path]:
    """
    迭代查找所有插件清单文件PLUGIN_MANIFEST_NAME=apecode_plugin.json
    
    查找逻辑：
    1. 对每个插件目录，先检查目录本身是否有清单
    2. 然后检查子目录是否有清单（一层嵌套）
    
    参数：
    - plugin_dirs: 插件目录列表
    
    返回：
    - 所有找到的清单文件路径列表
    """
    manifests: list[Path] = []
    for raw_dir in plugin_dirs:
        plugin_dir = raw_dir.expanduser().resolve()
        if not plugin_dir.exists() or not plugin_dir.is_dir():
            continue

        # 检查当前目录是否直接有清单
        direct_manifest = plugin_dir / PLUGIN_MANIFEST_NAME
        if direct_manifest.exists() and direct_manifest.is_file():
            manifests.append(direct_manifest)

        # 检查子目录是否有清单
        for child in sorted(plugin_dir.iterdir()):
            if not child.is_dir():
                continue
            nested_manifest = child / PLUGIN_MANIFEST_NAME
            if nested_manifest.exists() and nested_manifest.is_file():
                manifests.append(nested_manifest)
    return manifests


@dataclass(frozen=True, slots=True)
class PluginToolSpec:
    """
    插件工具定义 - 从清单中解析出来的工具规范
    
    字段：
    - plugin_name: 所属插件名称
    - name: 工具名称
    - description: 工具描述
    - parameters: JSON Schema 参数定义
    - mutating: 是否是变异操作（需要审批）
    - timeout_sec: 超时时间（秒）
    - command: shell 命令字符串（和 argv 二选一）
    - argv: 参数列表（和 command 二选一）
    - workdir: 执行命令的工作目录（插件目录）
    """

    plugin_name: str
    name: str
    description: str
    parameters: dict[str, Any]
    mutating: bool
    timeout_sec: int
    command: str | None = None
    argv: list[str] | None = None
    workdir: Path | None = None


@dataclass(frozen=True, slots=True)
class PluginCommandSpec:
    """
    插件斜杠命令定义
    
    字段：
    - plugin_name: 所属插件名称
    - name: 命令名称
    - description: 命令描述
    - usage: 使用示例
    - output: 执行时显示的输出
    - agent_input_template: 给 Agent 的输入模板（支持 {args} 占位符）
    """

    plugin_name: str
    name: str
    description: str
    usage: str
    output: str
    agent_input_template: str | None = None


@dataclass(slots=True)
class PluginLoadResult:
    """
    插件加载结果 - 包含加载的所有内容和错误
    
    字段：
    - tool_names: 加载的工具名称列表
    - commands: 加载的命令列表
    - skills: 加载的技能列表
    - errors: 加载过程中的错误信息
    """

    tool_names: list[str] = field(default_factory=list)
    commands: list[PluginCommandSpec] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _ParsedManifest:
    """
    解析后的清单内部结构（不对外暴露）
    
    包含从清单文件中解析出的所有内容
    """
    plugin_name: str
    tools: list[PluginToolSpec]
    commands: list[PluginCommandSpec]
    skills: list[Skill]


def _parse_tools(payload: dict[str, Any], *, manifest_path: Path, plugin_name: str) -> list[PluginToolSpec]:
    """
    解析清单中的 tools 字段
    
    参数：
    - payload: 清单 JSON 数据
    - manifest_path: 清单文件路径（用于确定 workdir）
    - plugin_name: 插件名称
    
    返回：
    - 工具规范列表
    
    抛出：
    - ValueError: 格式错误时
    """
    raw_tools = payload.get("tools", [])
    if not isinstance(raw_tools, list):
        raise ValueError("`tools` must be a list")

    tools: list[PluginToolSpec] = []
    for item in raw_tools:
        if not isinstance(item, dict):
            raise ValueError("tool entry must be an object")

        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("tool.name is required")

        description = str(item.get("description", "")).strip() or f"Plugin tool `{name}`"
        parameters = item.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("tool.parameters must be an object")

        # 命令可以是 shell 字符串或参数列表，二选一
        command = str(item.get("command", "")).strip() or None
        argv_value = item.get("argv")
        argv = [str(value) for value in argv_value] if isinstance(argv_value, list) else None
        if not command and not argv:
            raise ValueError(f"tool `{name}` must provide `command` or `argv`")
        if command and argv:
            raise ValueError(f"tool `{name}` cannot provide both `command` and `argv`")

        timeout_sec = max(1, min(int(item.get("timeout_sec", 120)), 1800))
        tools.append(
            PluginToolSpec(
                plugin_name=plugin_name,
                name=name,
                description=description,
                parameters=parameters,
                mutating=bool(item.get("mutating", False)),# 这里如果没有指定，默认是 False
                timeout_sec=timeout_sec,
                command=command,
                argv=argv,
                workdir=manifest_path.parent,
            )
        )
    return tools


def _parse_commands(payload: dict[str, Any], *, plugin_name: str) -> list[PluginCommandSpec]:
    """
    解析清单中的 commands 字段
    
    参数：
    - payload: 清单 JSON 数据
    - plugin_name: 插件名称
    
    返回：
    - 命令规范列表
    """
    raw_commands = payload.get("commands", [])
    if not isinstance(raw_commands, list):
        raise ValueError("`commands` must be a list")

    commands: list[PluginCommandSpec] = []
    for item in raw_commands:
        if not isinstance(item, dict):
            raise ValueError("command entry must be an object")
        raw_name = str(item.get("name", "")).strip()
        if not raw_name:
            raise ValueError("command.name is required")
        name = _sanitize_name(raw_name)
        description = str(item.get("description", "")).strip() or f"Plugin command `{name}`"
        usage = str(item.get("usage", "")).strip() or f"/{name} [args]"
        #这里插件工具的command只是把command字段的值直接作为输出，没有其他处理，不想command里面有具体的handler
        output = str(item.get("output", "")).strip() or f"Running plugin command `/{name}`..."
        template = str(item.get("agent_input_template", "")).strip() or None
        commands.append(
            PluginCommandSpec(
                plugin_name=plugin_name,
                name=name,
                description=description,
                usage=usage,
                output=output,
                agent_input_template=template,
            )
        )
    return commands


def _parse_skills(payload: dict[str, Any], *, manifest_path: Path, plugin_name: str) -> list[Skill]:
    """
    解析清单中的 skills 字段
    
    技能内容可以是：
    1. 内联的 content 字段
    2. 相对插件目录的 file 路径
    
    参数：
    - payload: 清单 JSON 数据
    - manifest_path: 清单文件路径（用于解析相对路径）
    - plugin_name: 插件名称
    
    返回：
    - 技能列表
    """
    raw_skills = payload.get("skills", [])
    if not isinstance(raw_skills, list):
        raise ValueError("`skills` must be a list")

    skills: list[Skill] = []
    for item in raw_skills:
        if not isinstance(item, dict):
            raise ValueError("skill entry must be an object")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("skill.name is required")

        description = str(item.get("description", "")).strip()
        #内联内容，即在配置文件中直接把skill的content的内容写好了，不去用file字段
        inline_content = item.get("content")
        file_path = str(item.get("file", "")).strip()

        # 优先使用内联内容，其次使用文件
        content_text: str
        path: Path | None = None
        if isinstance(inline_content, str) and inline_content.strip():
            content_text = inline_content.strip()
        elif file_path:
            path = (manifest_path.parent / file_path).resolve()
            if not path.exists() or not path.is_file():
                raise ValueError(f"skill `{name}` file not found: {file_path}")
            content_text = path.read_text(encoding="utf-8", errors="replace").strip()
        else:
            raise ValueError(f"skill `{name}` requires `content` or `file`")

        # 从内容中提取描述（如果未提供）
        derived_description = "No description."
        for raw_line in content_text.splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                derived_description = line[:160]
                break

        skills.append(
            Skill(
                name=name,
                description=description or derived_description,
                path=path,
                inline_content=None if path is not None else content_text,
                source=f"plugin:{plugin_name}",
            )
        )
    return skills


def _parse_manifest(manifest_path: Path) -> _ParsedManifest:
    """
    解析单个插件清单文件
    
    步骤：
    1. 读取并解析 JSON
    2. 提取插件名称（默认用目录名）
    3. 解析 tools、commands、skills
    
    参数：
    - manifest_path: 清单文件路径
    
    返回：
    - 解析后的清单对象
    
    抛出：
    - Exception: 任何解析错误
    """
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    plugin_name = str(payload.get("name", manifest_path.parent.name)).strip() or manifest_path.parent.name
    return _ParsedManifest(
        plugin_name=plugin_name,
        tools=_parse_tools(payload, manifest_path=manifest_path, plugin_name=plugin_name),
        commands=_parse_commands(payload, plugin_name=plugin_name),
        skills=_parse_skills(payload, manifest_path=manifest_path, plugin_name=plugin_name),
    )


def _build_tool_handler(spec: PluginToolSpec):
    """
    为插件工具构建处理器函数
    
    工作原理：
    对于注册的tool，根据spec即插件清单中的工具规范，构建一个工具处理器函数，该函数负责：
    1. 将工具参数通过 stdin 以 JSON 格式传给命令行工具
    2. 读取命令行工具的 stdout 作为工具结果
    3. 处理超时和错误
    
    参数：
    - spec: 工具规范
    
    返回：
    - 可以注册到 ToolRegistry 的 handler 函数
    """
    def _handler(_ctx: ToolContext, args: dict[str, Any]) -> str:
        if spec.argv:
            command: str | list[str] = spec.argv
            shell = False
        else:
            command = spec.command or ""
            shell = True

        # 通过subprocess.run执行外部命令，参数通过 stdin JSON 传递
        proc = subprocess.run(
            command,
            shell=shell,
            cwd=spec.workdir,
            input=json.dumps(args, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=spec.timeout_sec,
            check=False,
        )
        output = (proc.stdout or "").strip()
        error_text = (proc.stderr or "").strip()
        if proc.returncode != 0:
            detail = error_text or output or f"exit_code={proc.returncode}"
            return f"plugin `{spec.plugin_name}` tool `{spec.name}` failed: {detail}"
        if not output:
            return f"plugin `{spec.plugin_name}` tool `{spec.name}` finished with empty output"
        if len(output) > 8000:
            return output[:8000] + "\n... (truncated)"
        return output

    return _handler


def load_plugins(registry: ToolRegistry, plugin_dirs: list[Path]) -> PluginLoadResult:
    """
    加载插件并注册声明式插件工具
    
    这是模块的主入口函数，执行流程：
    1. 查找所有清单文件
    2. 逐个解析清单
    3. 注册工具（带命名空间，防止冲突）
    4. 收集命令和技能
    5. 记录错误但继续处理其他插件
    
    参数：
    - registry: 工具注册表（用于注册插件工具）
    - plugin_dirs: 插件目录列表
    
    返回：
    - 包含所有加载内容和错误的结果对象
    """
    result = PluginLoadResult()
    existing_tool_names = set(registry.list_tool_names())

    for manifest in _iter_manifest_files(plugin_dirs):
        try:
            parsed = _parse_manifest(manifest)
        except Exception as exc:
            result.errors.append(f"invalid plugin manifest `{manifest}`: {exc}")
            continue

        # 注册工具（用命名空间避免冲突）
        for spec in parsed.tools:
            namespaced = f"{_sanitize_name(spec.plugin_name)}__{_sanitize_name(spec.name)}"
            if namespaced in existing_tool_names:
                result.errors.append(f"duplicate plugin tool ignored: {namespaced}")
                continue
            registry.register(
                Tool(
                    name=namespaced,
                    description=f"[plugin:{spec.plugin_name}] {spec.description}",
                    parameters=spec.parameters,
                    handler=_build_tool_handler(spec),
                    mutating=spec.mutating,
                )
            )
            existing_tool_names.add(namespaced)
            result.tool_names.append(namespaced)

        # 收集命令和技能（暂不注册，由调用方处理）
        result.commands.extend(parsed.commands)
        result.skills.extend(parsed.skills)

    return result
