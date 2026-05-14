"""
CLI 入口模块 - ApeCode 项目的主入口

职责：
1. 解析命令行参数
2. 加载 .env 配置文件
3. 组装所有运行时组件（Agent、工具、命令等）
4. 运行交互式 REPL 或单次执行模式
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from apecode import __version__
from apecode.agent import AgentCallbacks, AgentConfig, NanoCodeAgent
from apecode.commands import (
    CommandRegistry,
    create_default_commands,
    create_template_command,
)
from apecode.console import (
    InputSession,
    ask_approval,
    console,
    print_agent,
    print_error,
    print_plan,
    print_status,
    print_thinking,
    print_tool_call,
    print_tool_result,
    set_status,
)
from apecode.mcp import McpBridge, load_mcp_tools
from apecode.model_adapters import ModelError, create_model_client
from apecode.plugins import load_plugins
from apecode.skills import SkillCatalog
from apecode.subagents import SubagentProxy, SubagentRunner
from apecode.system_prompt import build_system_prompt
from apecode.tools import (
    ApprovalPolicy,
    SandboxMode,
    ToolContext,
    create_default_registry,
)


@dataclass(slots=True)
class AppRuntime:
    """
    应用运行时容器 - 把所有组件组装在一起
    
    包含：
    - agent: 核心 AI 代理
    - commands: 斜杠命令注册表
    - mcp_bridge: MCP 工具桥接（可选）
    """

    agent: NanoCodeAgent
    commands: CommandRegistry
    mcp_bridge: McpBridge | None = None

    def close(self) -> None:
        """清理资源，关闭 MCP 连接"""
        if self.mcp_bridge is not None:
            self.mcp_bridge.close()


def _version_callback(value: bool) -> None:
    """版本号回调函数 - 当用户输入 --version/-V 时显示版本并退出"""
    if value:
        console.print(f"apecode {__version__}")
        raise typer.Exit()


def _approval_prompt(yolo_state: dict[str, bool], action: str, preview: str) -> bool:
    """
    审批提示函数
    
    参数：
    - yolo_state: yolo 模式状态（是否跳过确认）
    - action: 操作名称
    - preview: 操作预览
    
    返回：是否允许执行
    """
    if yolo_state["enabled"]:
        return True  # yolo 模式直接通过
    result = ask_approval(action, preview)
    if result and preview == "":
        pass
    return result


def _collect_skill_roots(cwd: Path, arg_values: list[str]) -> list[Path]:
    """
    收集技能根目录
    
    包含：
    1. 用户指定的目录（命令行参数）
    2. 当前工作目录下的 skills/ 目录
    """
    roots = [Path(item).expanduser() for item in arg_values]
    roots.append(cwd / "skills")
    return roots


def _collect_mcp_configs(cwd: Path, arg_values: list[str]) -> list[Path]:
    """
    收集 MCP 配置文件路径
    
    包含：
    1. 用户指定的配置文件
    2. 当前目录下的 .mcp.json
    3. 当前目录下的 apecode_mcp.json
    """
    paths = [Path(item).expanduser() for item in arg_values]
    paths.append(cwd / ".mcp.json")
    paths.append(cwd / "apecode_mcp.json")
    return paths


def _register_plugin_commands(runtime_commands: CommandRegistry, plugin_commands) -> tuple[int, list[str]]:
    """
    注册插件提供的命令
    
    返回：(加载成功数量, 错误列表)
    """
    loaded = 0
    errors: list[str] = []
    for spec in plugin_commands:
        try:
            runtime_commands.register(
                create_template_command(
                    name=spec.name,
                    description=f"[plugin:{spec.plugin_name}] {spec.description}",
                    usage=spec.usage,
                    output=spec.output,
                    agent_input_template=spec.agent_input_template,
                )
            )
            loaded += 1
        except ValueError as exc:
            errors.append(str(exc))
    return loaded, errors


def _make_callbacks(tool_context: ToolContext, *, indent: str = "") -> AgentCallbacks:
    """
    构建显示回调函数
    
    这些回调负责：
    - 显示状态（"思考中..."）
    - 显示思考过程
    - 显示工具调用
    - 显示工具结果
    
    参数：
    - indent: 缩进（用于子代理嵌套显示）
    """

    def _on_tool_result(name: str, result: str) -> None:
        """工具结果回调 - 特殊处理 update_plan 的显示"""
        if name == "update_plan":
            print_plan(tool_context.plan)
        else:
            print_tool_result(name, result)

    return AgentCallbacks(
        on_status=set_status,
        on_thinking=lambda text: print_thinking(text),
        on_tool_call=lambda name, args: print_tool_call(name, args),
        on_tool_result=_on_tool_result,
    )


def _build_runtime(
    *,
    provider: str,
    model: str,
    max_steps: int,
    timeout: int,
    temperature: float | None,
    cwd: Path,
    sandbox_mode: SandboxMode,
    approval_policy: ApprovalPolicy,
    yolo: bool,
    plugin_dirs: list[str],
    mcp_configs: list[str],
    skill_dirs: list[str],
) -> AppRuntime:
    """
    【核心函数】组装应用运行时
    
    这是整个程序最关键的函数，负责按顺序初始化和组装所有组件：
    
    步骤：
    1. 初始化工具上下文（沙箱、审批策略）
    2. 注册内置工具
    3. 加载插件
    4. 加载 MCP 工具
    5. 加载技能
    6. 构建系统提示
    7. 创建模型客户端
    8. 设置子代理
    9. 创建命令注册表
    10. 组装 NanoCodeAgent
    
    返回：组装好的 AppRuntime 对象
    """
    # 步骤 1: 处理 yolo 模式和工作目录
    # YOLO 模式 是 "You Only Live Once" 的缩写，意思是 直接执行，不需要确认 。
    # - 当启用 --yolo 选项时，Agent 在执行任何操作前都 不需要用户手动确认
    # - 相当于把 --approval-policy 设置为 always
    # - 所有操作直接放行，适合完全信任 Agent 的场景
    if yolo:
        approval_policy = ApprovalPolicy.ALWAYS

    # 步骤 1: 处理工作目录，把用户输入的路径转换为绝对路径，避免相对路径问题
    cwd = cwd.expanduser().resolve()
    yolo_state = {"enabled": yolo}
    
    # 步骤 2: 创建上下文（沙箱 + 审批策略）
    # tool_context 是一个上下文对象，包含了工具、审批策略、沙箱模式、执行计划等信息
    tool_context = ToolContext(
        cwd=cwd,
        ask_approval=lambda action, preview: _approval_prompt(yolo_state, action, preview),
        sandbox_mode=sandbox_mode,
        approval_policy=approval_policy,
    )
    
    # 步骤 3: 注册内置工具
    tools = create_default_registry(tool_context)
    
    # 步骤 4: 加载插件（工具、命令、技能）
    plugin_dir_paths = [Path(item) for item in plugin_dirs]
    plugin_dir_paths.append(cwd / "plugins")
    # 这里load_plugins会注册插件配置文件提供的tool，skill和command，但是skill和command暂时还是解析，只有tool是真正regist到tools里了
    # skill和command是后续再处理的
    plugin_result = load_plugins(tools, plugin_dir_paths)
    if plugin_result.tool_names:
        print_status(f"[plugin] loaded {len(plugin_result.tool_names)} tools")
    for error in plugin_result.errors:
        print_error(f"[plugin] {error}")

    # 步骤 5: 加载 MCP 工具
    mcp_config_paths = _collect_mcp_configs(cwd, mcp_configs)
    mcp_bridge = load_mcp_tools(tools, mcp_config_paths)
    if mcp_bridge.tool_names:
        print_status(f"[mcp] loaded {len(mcp_bridge.tool_names)} tools")
    for error in mcp_bridge.errors:
        print_error(f"[mcp] {error}")

    # 步骤 6: 加载skill
    skill_roots = _collect_skill_roots(cwd, skill_dirs)
    #从skill的目录中加载skill.md文件，构建SkillCatalog
    skills = SkillCatalog.from_roots(skill_roots)
    # 合并plugin提供的skill到SkillCatalog
    if plugin_result.skills:
        initial_count = len(skills.list_skills())
        skills = skills.with_additional(plugin_result.skills)
        merged_count = len(skills.list_skills()) - initial_count
        if merged_count > 0:
            print_status(f"[plugin] loaded {merged_count} skills")
    
    # 步骤 7: 获取当前目录列表，用于构建系统提示
    _dir_entries: list[str] = []
    try:
        for item in sorted(cwd.iterdir()):
            _dir_entries.append(f"{item.name}{'/' if item.is_dir() else ''}")
    except OSError:
        pass
    _dir_listing = "\n".join(_dir_entries) if _dir_entries else None
    base_prompt = build_system_prompt(cwd, skills_overview=skills.format_for_system_prompt(), dir_listing=_dir_listing)

    # 步骤 8: 创建模型客户端（OpenAI/Anthropic/Kimi）
    model_client = create_model_client(
        provider=provider,
        model=model,
        timeout=timeout,
        temperature=temperature,
    )

    # 步骤 9: 设置子代理（带缩进的回调，用于嵌套显示）
    sub_callbacks = _make_callbacks(tool_context, indent="    ")
    subagents = SubagentProxy(
        SubagentRunner(
            model=model_client,
            parent_tools=tools,
            base_system_prompt=base_prompt,
            max_steps=min(8, max(2, max_steps)),  # 子代理限制步数：2-8 步
            callbacks=sub_callbacks,
        )
    )
    
    # 步骤 10: 创建命令注册表
    commands = create_default_commands(tools=tools, skills=skills, subagents=subagents)
    loaded_command_count, command_errors = _register_plugin_commands(commands, plugin_result.commands)
    if loaded_command_count > 0:
        print_status(f"[plugin] loaded {loaded_command_count} commands")
    for error in command_errors:
        print_error(f"[plugin] {error}")
    
    # 步骤 11: 组装所有组件并返回
    return AppRuntime(
        agent=NanoCodeAgent(
            model=model_client,
            tools=tools,
            system_prompt=base_prompt,
            config=AgentConfig(max_steps=max(1, max_steps)),
            callbacks=_make_callbacks(tool_context),
        ),
        commands=commands,
        mcp_bridge=mcp_bridge,
    )


def _execute_agent_turn(agent: NanoCodeAgent, text: str) -> tuple[bool, str]:
    """
    执行一次 Agent 交互
    
    返回：(是否成功, 结果内容)
    """
    try:
        return True, agent.run(text)
    except (ModelError, RuntimeError) as exc:
        return False, str(exc)


def _run_repl(runtime: AppRuntime) -> int:
    """
    【REPL 主循环】交互式会话
    
    流程：
    1. 显示欢迎信息
    2. 进入循环：
       - 等待用户输入
       - 尝试处理斜杠命令
       - 如果不是命令，交给 Agent 处理
    """
    print_status("ApeCode nano agent. Type /exit to quit. Alt+Enter for multi-line.")
    command_names = [cmd.name for cmd in runtime.commands.list_commands()]
    print_status(f"Available commands: {', '.join(command_names)}")
    session = InputSession(command_names=command_names)
    
    while True:
        try:
            user_input = session.prompt()
        except (KeyboardInterrupt, EOFError):
            console.print()
            return 0
        print_status(f"User input: {user_input}")
        if not user_input:
            continue

        # 先尝试处理斜杠命令（如 /help, /tools 等）
        # 必须以/开头，否则command_result为None
        command_result = runtime.commands.run(user_input)
        if command_result is not None:
            print_agent(command_result.output)
            if command_result.should_exit:
                return 0
            if command_result.agent_input is None:
                continue
            ok, output = _execute_agent_turn(runtime.agent, command_result.agent_input)
            if ok:
                print_agent(output)
            else:
                print_error(output)
            continue

        # 不是命令，交给 Agent 处理
        ok, output = _execute_agent_turn(runtime.agent, user_input)
        if ok:
            print_agent(output)
        else:
            print_error(output)


# 初始化 Typer CLI 应用
app = typer.Typer(
    name="ape",
    help="ApeCode - nano terminal code agent",
    add_completion=False,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    # 【一次性任务提示】如果为空，启动交互式 REPL 模式
    prompt: Annotated[
        list[str] | None,
        typer.Argument(help="One-shot prompt. If empty, starts REPL."),
    ] = None,
    # 【模型提供商】可选: openai/anthropic/kimi，可通过环境变量 APECODE_PROVIDER 设置
    provider: Annotated[str, typer.Option(envvar="APECODE_PROVIDER", help="Model provider.")] = "kimi",
    # 【模型名称】具体的模型 ID，可通过环境变量 APECODE_MODEL 设置
    model: Annotated[str, typer.Option(envvar="APECODE_MODEL", help="Model name.")] = "kimi-k2.5",
    # 【最大执行步数】Agent 循环的最大步骤数，防止无限循环
    max_steps: Annotated[int, typer.Option(help="Max agent loop steps.")] = 20,
    # 【请求超时】模型 API 请求的超时时间（秒）
    timeout: Annotated[int, typer.Option(help="Model request timeout in seconds.")] = 120,
    # 【温度参数】控制模型输出的随机性，None 表示使用提供商默认值
    temperature: Annotated[
        float | None,
        typer.Option(help="Model temperature. Provider default if omitted."),
    ] = None,
    # 【工作目录】设置工作空间目录，默认为当前目录
    cwd: Annotated[str, typer.Option(help="Workspace directory.")] = "",
    # 【沙箱模式】控制文件系统访问权限，可通过环境变量 APECODE_SANDBOX_MODE 设置
    #   - WORKSPACE_WRITE: 可读写工作区（默认）
    #   - WORKSPACE_READ: 只读工作区
    #   - UNRESTRICTED: 无限制
    sandbox_mode: Annotated[SandboxMode, typer.Option(envvar="APECODE_SANDBOX_MODE", help="Sandbox mode.")] = SandboxMode.WORKSPACE_WRITE,
    # 【审批策略】控制工具执行前是否需要用户确认，可通过环境变量 APECODE_APPROVAL_POLICY 设置
    #   - ON_REQUEST: 按需确认（默认）
    #   - ALWAYS: 总是需要确认
    #   - NEVER: 从不确认（全自动）
    approval_policy: Annotated[
        ApprovalPolicy,
        typer.Option(envvar="APECODE_APPROVAL_POLICY", help="Approval policy."),
    ] = ApprovalPolicy.ON_REQUEST,
    # 【插件目录】插件加载目录，可重复指定多个目录
    plugin_dir: Annotated[list[str] | None, typer.Option(help="Plugin directory (can be repeated).")] = None,
    # 【MCP 配置】MCP (Model Context Protocol) 配置 JSON 文件，可重复指定多个
    mcp_config: Annotated[
        list[str] | None,
        typer.Option(help="MCP config JSON file (can be repeated."),
    ] = None,
    # 【技能目录】技能模板根目录，可重复指定多个目录
    skill_dir: Annotated[
        list[str] | None,
        typer.Option(help="Skill root directory (can be repeated."),
    ] = None,
    # 【YOLO 模式】快捷方式，等同于 --approval-policy always（全自动不确认）
    yolo: Annotated[bool, typer.Option("--yolo", help="Shortcut for --approval-policy always.")] = False,
    # 【版本信息】显示版本号并退出
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version.",
        ),
    ] = None,
) -> None:
    """
    【程序主入口】ApeCode 命令行应用
    
    支持两种运行模式：
    1. 交互式 REPL（默认，不提供 prompt）
    2. 单次执行模式（提供 prompt 参数）
    
    配置优先级（从高到低）：
    1. 命令行参数
    2. 环境变量
    3. .env 文件
    4. 默认值
    """
    # 步骤 1: 加载 .env 配置文件
    workspace = Path(cwd) if cwd else Path.cwd()
    env_path = workspace / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # 如果当前目录没有 .env，尝试从项目根目录加载
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    # 步骤 2: 组装运行时
    runtime: AppRuntime | None = None
    try:
        runtime = _build_runtime(
            provider=provider,
            model=model,
            max_steps=max_steps,
            timeout=timeout,
            temperature=temperature,
            cwd=workspace,
            sandbox_mode=sandbox_mode,
            approval_policy=approval_policy,
            yolo=yolo,
            plugin_dirs=plugin_dir or [],
            mcp_configs=mcp_config or [],
            skill_dirs=skill_dir or [],
        )
    except RuntimeError as exc:
        print_error(f"ApeCode setup error: {exc}")
        raise typer.Exit(code=1) from None

    # 步骤 3: 根据是否提供 prompt 选择运行模式
    try:
        prompt_text = " ".join(prompt).strip() if prompt else ""
        
        # 模式 A: 没有提供 prompt - 进入交互式 REPL
        if not prompt_text:
            code = _run_repl(runtime)
            raise typer.Exit(code=code)

        # 模式 B: 提供了 prompt - 单次执行模式
        # 先尝试处理斜杠命令（如 /help, /tools 等）
        # 必须以/开头，否则command_result为None
        command_result = runtime.commands.run(prompt_text)
        if command_result is not None:
            print_agent(command_result.output)
            if command_result.should_exit:
                raise typer.Exit()
            if command_result.agent_input is None:
                raise typer.Exit()
            # 命令执行成功，将结果作为下一轮的 prompt
            # 这里主要针对skill的情况，因为只有skill会返回agent_input，内容为skill.md的具体内容
            prompt_text = command_result.agent_input

        ok, output = _execute_agent_turn(runtime.agent, prompt_text)
        if not ok:
            print_error(f"ApeCode runtime error: {output}")
            raise typer.Exit(code=2)
        print_agent(output)
    
    # 步骤 4: 清理资源（无论成功失败都执行）
    finally:
        if runtime is not None:
            runtime.close()
