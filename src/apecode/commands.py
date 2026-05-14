"""
交互式斜杠命令框架

职责：
- 定义命令元数据和执行结果
- 管理命令注册和查找
- 解析用户输入并执行对应命令
- 提供内置命令（/help, /tools, /skills, /exit 等）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from apecode.skills import SkillCatalog
from apecode.subagents import SubagentProxy
from apecode.tools import ToolRegistry

# 命令处理器类型：接收参数字符串，返回命令结果
CommandHandler = Callable[[str], "CommandResult"]


@dataclass(slots=True)
class CommandResult:
    """
    斜杠命令执行结果
    
    字段：
    - output: 显示给用户的输出文本
    - agent_input: 可选，传给 Agent 的输入（用于触发自动对话）,用于下一轮的prompt构建
    - should_exit: 是否退出会话
    """

    output: str
    agent_input: str | None = None
    should_exit: bool = False


@dataclass(slots=True)
class SlashCommand:
    """
    斜杠命令的元数据和处理器
    
    字段：
    - name: 命令名称（不含斜杠）
    - description: 命令描述
    - usage: 使用说明
    - handler: 命令处理函数
    """

    name: str
    description: str
    usage: str
    handler: CommandHandler


class CommandRegistry:
    """
    斜杠命令注册表
    
    功能：
    - 注册新命令
    - 查找命令
    - 列出所有命令
    - 解析用户输入并执行命令
    """

    def __init__(self) -> None:
        # 存储所有已注册的命令，键为命令名
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand, *, replace: bool = False) -> None:
        """
        注册一个新命令
        
        参数：
        - command: 要注册的 SlashCommand 对象
        - replace: 是否允许覆盖已存在的同名命令
        
        异常：
        - ValueError: 命令已存在且 replace=False 时抛出
        """
        if not replace and command.name in self._commands:
            raise ValueError(f"command already registered: /{command.name}")
        self._commands[command.name] = command

    def get(self, name: str) -> SlashCommand | None:
        """
        根据命令名获取命令
        
        参数：
        - name: 命令名
        
        返回：
        - SlashCommand 对象，找不到返回 None
        """
        return self._commands.get(name)

    def list_commands(self) -> list[SlashCommand]:
        """
        列出所有已注册的命令（按名称排序）
        
        返回：
        - SlashCommand 列表
        """
        return [self._commands[name] for name in sorted(self._commands)]

    def run(self, raw_input: str) -> CommandResult | None:
        """
        执行用户输入的斜杠命令
        
        参数：
        - raw_input: 用户原始输入字符串
        
        返回：
        - CommandResult: 是斜杠命令时返回执行结果
        - None: 不是斜杠命令时返回
        """
        # 检查是否以斜杠开头
        if not raw_input.startswith("/"):
            return None
        # 去掉开头的斜杠
        payload = raw_input[1:].strip()
        # 空命令提示
        if not payload:
            return CommandResult(output="Empty slash command. Use /help.")
        # 分离命令名和参数
        if " " in payload:
            name, args = payload.split(" ", 1)
            args = args.strip()
        else:
            name, args = payload, ""
        # 查找命令
        command = self.get(name)
        if command is None:
            return CommandResult(output=f"Unknown command: /{name}. Use /help.")
        # 执行命令处理器
        return command.handler(args)


def create_template_command(
    *,
    name: str,
    description: str,
    usage: str,
    output: str,
    agent_input_template: str | None = None,
) -> SlashCommand:
    """
    从简单文本模板构建斜杠命令
    主要是为了加载配置文件中的command命令，用于插件扩展
    
    参数：
    - name: 命令名
    - description: 命令描述
    - usage: 使用说明
    - output: 命令输出文本
    - agent_input_template: 可选，Agent 输入模板，可包含 {args} 占位符
    
    返回：
    - 构建好的 SlashCommand 对象
    """

    def _handler(args: str) -> CommandResult:
        agent_input: str | None = None
        if agent_input_template is not None:
            agent_input = agent_input_template.replace("{args}", args.strip())
        return CommandResult(output=output, agent_input=agent_input)

    return SlashCommand(
        name=name,
        description=description,
        usage=usage,
        handler=_handler,
    )


def create_default_commands(
    *,
    tools: ToolRegistry,
    skills: SkillCatalog,
    subagents: SubagentProxy | None = None,
) -> CommandRegistry:
    """
    构建内置斜杠命令
    
    参数：
    - tools: 工具注册表
    - skills: 技能目录
    - subagents: 子代理代理（可选）
    
    返回：
    - 包含所有内置命令的 CommandRegistry
    
    内置命令：
    - /help: 显示可用命令
    - /tools: 列出已注册的工具
    - /skills: 列出已发现的技能
    - /skill <name>: 运行指定技能
    - /plan: 显示当前任务计划
    - /subagents: 列出子代理配置
    - /delegate <task>: 委派任务给子代理
    - /exit: 退出会话
    """
    registry = CommandRegistry()

    def _help(_args: str) -> CommandResult:
        """
        /help 命令：显示所有可用命令
        """
        lines = ["Available commands:"]
        for command in registry.list_commands():
            lines.append(f"- /{command.name}: {command.description} ({command.usage})")
        return CommandResult(output="\n".join(lines))

    def _tools(_args: str) -> CommandResult:
        """
        /tools 命令：列出所有已注册的工具
        """
        names = tools.list_tool_names()
        if not names:
            return CommandResult(output="No tools registered.")
        return CommandResult(output="Tools:\n" + "\n".join(f"- {name}" for name in names))

    def _skills(_args: str) -> CommandResult:
        """
        /skills 命令：列出所有已发现的技能
        """
        records = skills.list_skills()
        if not records:
            return CommandResult(output="No skills found.")
        lines = ["Skills:"]
        for skill in records:
            lines.append(f"- {skill.name}: {skill.description}")
        return CommandResult(output="\n".join(lines))

    def _skill(args: str) -> CommandResult:
        """
        /skill <name> 命令：运行指定技能作为提示模板
        """
        if not args:
            return CommandResult(output="Usage: /skill <name> [extra request]")
        parts = args.split(" ", 1)
        name = parts[0].strip().lower()
        extra = parts[1].strip() if len(parts) > 1 else ""
        skill = skills.get(name)
        if skill is None:
            return CommandResult(output=f"Skill not found: {name}")
        body = skill.read_text()
        if extra:
            body = f"{body}\n\nUser request:\n{extra}"
        return CommandResult(output=f"Running skill `{name}`...", agent_input=body)

    def _plan(_args: str) -> CommandResult:
        """
        /plan 命令：显示当前内存中的任务计划
        """
        plan = tools.context.plan
        if not plan:
            return CommandResult(output="Plan is empty.")
        lines = ["Current plan:"]
        for item in plan:
            lines.append(f"- [{item['status']}] {item['step']}")
        return CommandResult(output="\n".join(lines))

    def _exit(_args: str) -> CommandResult:
        """
        /exit 命令：退出当前会话
        """
        return CommandResult(output="Bye.", should_exit=True)

    def _subagents(_args: str) -> CommandResult:
        """
        /subagents 命令：列出可用的子代理配置
        """
        if subagents is None:
            return CommandResult(output="Subagents are disabled.")
        profiles = subagents.list_profiles()
        lines = ["Subagent profiles:"]
        for profile in profiles:
            lines.append(f"- {profile['name']}: {profile['description']}")
        return CommandResult(output="\n".join(lines))

    def _delegate(args: str) -> CommandResult:
        """
        /delegate <task> 命令：委派子任务给子代理
        用法：/delegate [profile::] <task>
        """
        if subagents is None:
            return CommandResult(output="Subagents are disabled.")
        if not args.strip():
            return CommandResult(output="Usage: /delegate [profile::] <task>")
        # 解析 profile::task 格式
        if "::" in args:
            profile, task = args.split("::", 1)
            profile = profile.strip().lower()
            task = task.strip()
        else:
            profile, task = "general", args.strip()
        if not task:
            return CommandResult(output="Usage: /delegate [profile::] <task>")
        try:
            output = subagents.run(task=task, profile=profile)
        except ValueError as exc:
            return CommandResult(output=str(exc))
        except RuntimeError as exc:
            return CommandResult(output=f"Subagent execution failed: {exc}")
        return CommandResult(output=f"Subagent `{profile}`:\n{output}")

    # 注册 /help 命令
    registry.register(
        SlashCommand(
            name="help",
            description="Show available slash commands.",
            usage="/help",
            handler=_help,
        )
    )
    # 注册 /tools 命令
    registry.register(
        SlashCommand(
            name="tools",
            description="List currently registered tools.",
            usage="/tools",
            handler=_tools,
        )
    )
    # 注册 /skills 命令
    registry.register(
        SlashCommand(
            name="skills",
            description="List discovered skills.",
            usage="/skills",
            handler=_skills,
        )
    )
    # 注册 /skill 命令
    registry.register(
        SlashCommand(
            name="skill",
            description="Run one skill as a prompt template.",
            usage="/skill <name> [extra request]",
            handler=_skill,
        )
    )
    # 注册 /plan 命令
    registry.register(
        SlashCommand(
            name="plan",
            description="Print the latest in-memory plan.",
            usage="/plan",
            handler=_plan,
        )
    )
    # 注册 /subagents 命令
    registry.register(
        SlashCommand(
            name="subagents",
            description="List available subagent profiles.",
            usage="/subagents",
            handler=_subagents,
        )
    )
    # 注册 /delegate 命令
    registry.register(
        SlashCommand(
            name="delegate",
            description="Delegate a focused sub-task to a subagent.",
            usage="/delegate [profile::] <task>",
            handler=_delegate,
        )
    )
    # 注册 /exit 命令
    registry.register(
        SlashCommand(
            name="exit",
            description="Exit current session.",
            usage="/exit",
            handler=_exit,
        )
    )
    return registry
