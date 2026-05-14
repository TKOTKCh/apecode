"""
ApeCode 子代理委托模块 - 简化的子代理执行

核心功能：
- 定义子代理角色配置（SubagentProfile）
- 提供子代理运行器（SubagentRunner），使用隔离的只读工具环境
- 提供子代理代理器（SubagentProxy），用于与斜杠命令解耦

内置子代理角色：
- general: 通用目的委托，专注执行任务
- reviewer: 代码审查，识别 bug/风险
- researcher: 研究代码库上下文并总结发现
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apecode.agent import AgentCallbacks, AgentConfig, ChatModel, NanoCodeAgent
from apecode.tools import ToolContext, ToolRegistry


@dataclass(frozen=True, slots=True)
class SubagentProfile:
    """
    子代理使用的角色提示词配置
    
    字段：
    - name: 角色名称（用于选择）
    - description: 角色描述
    - prompt: 角色提示词内容
    """

    name: str
    description: str
    prompt: str


DEFAULT_SUBAGENT_PROFILES: tuple[SubagentProfile, ...] = (
    # 通用子代理：用于执行各种具体子任务，保持回答简洁，报告具体结果
    # 适用场景：数据收集、简单分析、快速任务等
    SubagentProfile(
        name="general",
        description="General-purpose delegate for focused task execution.",
        prompt=("You are a delegated helper agent. Focus only on the assigned sub-task, keep answers concise, and report concrete results."),
    ),

    
    # 代码审查子代理：专注于审查代码变更，识别 bug 和风险
    # 优先关注：代码正确性、回归问题、缺失的测试
    # 输出格式：先列出发现的问题，再提供简短总结
    SubagentProfile(
        name="reviewer",
        description="Review code changes and identify bugs/risks.",
        prompt=("You are a code reviewer subagent. Prioritize correctness, regressions, and missing tests. Provide findings first, then a short summary."),
    ),

    # 研究子代理：深入检查代码库上下文，总结发现
    # 工作方式：从文件和工具中收集高价值事实，清晰陈述假设
    # 输出格式：结构化的总结报告
    SubagentProfile(
        name="researcher",
        description="Inspect codebase context and summarize findings.",
        prompt=("You are a research subagent. Gather high-signal facts from files/tools, state assumptions clearly, and return a structured summary."),
    ),
    
)


class SubagentRunner:
    """
    子代理运行器 - 使用隔离的只读工具运行时执行委托任务
    
    特性：
    - 只读模式：子代理只能使用非变异工具
    - 隔离环境：子代理有独立的工具上下文
    - 可配置：支持自定义角色配置、最大步数等
    """

    def __init__(
        self,
        *,
        model: ChatModel,
        parent_tools: ToolRegistry,
        base_system_prompt: str,
        max_steps: int = 8,
        profiles: tuple[SubagentProfile, ...] = DEFAULT_SUBAGENT_PROFILES,
        callbacks: AgentCallbacks | None = None,
    ) -> None:
        """
        初始化子代理运行器
        
        参数：
        - model: 聊天模型
        - parent_tools: 父代理的工具注册表（从中选择非变异工具）
        - base_system_prompt: 基础系统提示词
        - max_steps: 最大执行步数
        - profiles: 子代理角色配置元组
        - callbacks: 代理回调（可选）
        """
        self._model = model
        self._parent_tools = parent_tools
        self._base_system_prompt = base_system_prompt
        self._max_steps = max(1, max_steps)
        self._profiles = {profile.name: profile for profile in profiles}
        self._callbacks = callbacks

    def list_profiles(self) -> list[SubagentProfile]:
        """
        列出所有可用的子代理角色
        
        返回：
        - 按名称排序的角色配置列表
        """
        return [self._profiles[name] for name in sorted(self._profiles)]

    def run(self, *, task: str, profile: str = "general") -> str:
        """
        运行子代理执行任务
        
        参数：
        - task: 子任务描述
        - profile: 使用的角色名称（默认 "general"）
        
        返回：
        - 子代理执行结果字符串
        
        异常：
        - ValueError: 未知角色或任务为空
        """
        profile_obj = self._profiles.get(profile)
        if profile_obj is None:
            raise ValueError(f"unknown subagent profile: {profile}")
        if not task.strip():
            raise ValueError("task cannot be empty")

        # 构建子代理的工具环境
        tools = self._build_subagent_tools()
        agent = NanoCodeAgent(
            model=self._model,
            tools=tools,
            system_prompt=(f"{self._base_system_prompt}\n\n# Subagent profile: {profile_obj.name}\n{profile_obj.prompt}\n"),
            config=AgentConfig(max_steps=self._max_steps),
            callbacks=self._callbacks,
        )
        return agent.run(task)

    def _build_subagent_tools(self) -> ToolRegistry:
        """
        构建子代理的工具注册表（只读模式）
        
        过滤规则：
        - 跳过所有变异工具（mutating=True）
        - 跳过 update_plan 工具
        - sandbox_mode 设置为 "read-only"
        - approval_policy 设置为 "never"
        
        返回：
        - 子代理的工具注册表
        """
        parent_context = self._parent_tools.context
        sub_context = ToolContext(
            cwd=parent_context.cwd,
            ask_approval=parent_context.ask_approval,
            sandbox_mode="read-only",
            approval_policy="never",
            plan=[],
        )
        registry = ToolRegistry(sub_context)
        for tool in self._parent_tools.list_tools():
            if tool.mutating:
                continue
            if tool.name in {"update_plan"}:
                continue
            registry.register(tool)
        return registry


class SubagentProxy:
    """
    子代理代理器 - 用于斜杠命令，实现依赖解耦的小适配器
    
    提供简化的接口给斜杠命令使用，避免直接依赖 SubagentRunner
    """

    def __init__(self, runner: SubagentRunner):
        self._runner = runner

    def list_profiles(self) -> list[dict[str, Any]]:
        """
        列出可用的角色（返回字典格式）
        
        返回：
        - 角色字典列表，每个字典包含 name 和 description
        """
        return [{"name": profile.name, "description": profile.description} for profile in self._runner.list_profiles()]

    def run(self, *, task: str, profile: str = "general") -> str:
        """
        运行子代理（代理方法）
        
        参数：
        - task: 子任务描述
        - profile: 使用的角色名称（默认 "general"）
        
        返回：
        - 子代理执行结果字符串
        """
        return self._runner.run(task=task, profile=profile)
