"""
终端输入输出工具模块

职责：
- 使用 Rich 进行漂亮的终端输出
- 使用 prompt_toolkit 进行交互式输入
- 显示工具调用、思考过程、执行状态
- 斜杠命令自动补全
"""

from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# 全局 Rich Console 对象（所有输出都通过它）
console = Console()

# ── Rich 输出辅助函数 ─────────────────────────────────────────────


def print_agent(text: str) -> None:
    """
    显示 Agent 的回答（Markdown 格式 + 绿色边框）
    
    在一个带绿色边框的面板里渲染 Markdown，标题是 "ape"
    """
    console.print(Panel(Markdown(text), title="ape", border_style="green"))


def print_error(text: str) -> None:
    """
    显示错误消息（红色粗体）
    
    格式：error> 你的错误信息
    """
    console.print(f"[bold red]error>[/bold red] {text}")


def print_status(text: str) -> None:
    """
    显示状态信息（灰色）
    
    如果是空字符串就不显示
    """
    if text:
        console.print(f"[dim]{text}[/dim]")


# ── 动态旋转指示器（Thinking...）────────────────────────────────────

# 全局变量，保存当前激活的旋转指示器
_active_spinner = None


def set_status(text: str) -> None:
    """
    开始或停止旋转指示器
    
    参数：
    - text: 显示的文本（空字符串表示停止）
    
    例子：
    - set_status("Thinking...") → 开始旋转
    - set_status("") → 停止旋转
    """
    global _active_spinner
    # 如果已有旋转指示器，先停止它
    if _active_spinner is not None:
        _active_spinner.__exit__(None, None, None)
        _active_spinner = None
    # 如果有文本，开始新的旋转指示器
    if text:
        _active_spinner = console.status(f"[bold green]{text}[/bold green]")
        _active_spinner.__enter__()


def ask_approval(action: str, preview: str) -> bool:
    """
    请求用户批准（针对会修改文件的工具调用）
    
    显示一个黄色边框的面板，然后问用户是否同意
    
    参数：
    - action: 操作名称（如 "write_file"）
    - preview: 操作预览（JSON 格式）
    
    返回：
    - True: 用户批准
    - False: 用户拒绝
    
    支持的回答：
    - y/yes: 同意
    - n/no: 拒绝（默认）
    - a/always: 总是同意（本次会话）
    """
    console.print(Panel(preview, title=f"[yellow]approve: {action}[/yellow]", border_style="yellow"))
    answer = console.input("[yellow]Approve? [y/N/a=always][/yellow] ").strip().lower()
    if answer == "a":
        return True  # 调用方处理 "always" 状态
    return answer in {"y", "yes", "a"}


@contextmanager
def status_spinner(text: str = "Thinking...") -> Generator[None]:
    """
    旋转指示器的上下文管理器
    
    使用方式：
        with status_spinner("思考中..."):
            ... 执行耗时操作 ...
    
    退出上下文时自动停止旋转
    """
    with console.status(f"[bold green]{text}[/bold green]"):
        yield


# ── Agent event display ──────────────────────────────────────────────


def _extract_key_arg(name: str, arguments_json: str) -> str:
    """
    从工具调用参数中提取最关键的信息用于一行显示
    
    参数：
    - name: 工具名称
    - arguments_json: 工具参数（JSON 格式）
    
    返回：
    - 提取的关键信息字符串（截断到60字符）
    
    优先提取的字段：path, command, pattern, plan, content
    """
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return ""
    if not isinstance(args, dict):
        return ""
    # 优先提取的关键字段 —— 先匹配到的优先显示
    for key in ("path", "command", "pattern", "plan", "content"):
        if key in args:
            val = args[key]
            if isinstance(val, str):
                return val[:60]
            if isinstance(val, list):
                return f"[{len(val)} items]"
    # 如果没有匹配到，回退到第一个字符串值
    for val in args.values():
        if isinstance(val, str):
            return val[:60]
    return ""


def print_tool_call(name: str, arguments_json: str) -> None:
    """
    显示工具调用信息
    
    格式：> 工具名 (关键参数)
    """
    key_arg = _extract_key_arg(name, arguments_json)
    suffix = f" [dim]({key_arg})[/dim]" if key_arg else ""
    console.print(f"  [bold blue]> {name}[/bold blue]{suffix}")


def print_tool_result(name: str, result: str) -> None:
    """
    显示工具执行结果的简短预览
    
    根据结果内容自动判断是成功还是失败
    """
    is_error = result.startswith(("blocked by", "Rejected", "Unknown tool", "Tool execution failed"))
    marker = "[red]x[/red]" if is_error else "[green]ok[/green]"
    # 收集前几行有意义的内容用于预览
    lines: list[str] = []
    for line in result.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # 跳过元数据行，如 "exit_code=0"
        if stripped.startswith("exit_code="):
            if stripped != "exit_code=0":
                is_error = True
                marker = "[red]x[/red]"
            continue
        lines.append(stripped)
        if len(lines) >= 3:
            break
    preview = " | ".join(lines) if lines else "(empty)"
    if len(preview) > 120:
        preview = preview[:117] + "..."
    console.print(f"  {marker} [dim]{preview}[/dim]")


def print_thinking(text: str) -> None:
    """
    显示模型的思考/推理过程（灰色斜体）
    
    如果内容太长会自动截断，保持终端整洁
    """
    # 截断过长的思考内容，保持终端整洁
    lines = text.strip().splitlines()
    shown = [*lines[:3], "...", *lines[-2:]] if len(lines) > 6 else lines
    body = "\n".join(shown)
    console.print(f"[dim italic]{body}[/dim italic]")


def print_plan(plan: list[dict[str, str]]) -> None:
    """
    显示任务计划，带状态标记
    
    状态标记：
    - ~: 已完成（绿色，带删除线）
    - >: 进行中（青色）
    - -: 待执行（灰色）
    """
    if not plan:
        return
    for item in plan:
        step = item.get("step", "")
        status = item.get("status", "")
        if status == "completed":
            console.print(f"  [green]~[/green] [strike dim]{step}[/strike dim]")
        elif status == "in_progress":
            console.print(f"  [cyan]>[/cyan] {step}")
        else:
            console.print(f"  [dim]-[/dim] [dim]{step}[/dim]")


# ── 斜杠命令自动补全 ────────────────────────────────────────────────
class _SlashCompleter(Completer):
    """
    斜杠命令自动补全器
    
    当用户输入 "/" 开头的命令时，按 Tab 键可以自动补全
    """

    def __init__(self, command_names: Sequence[str]) -> None:
        self._names = sorted(command_names)

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        """
        生成补全建议
        
        参数：
        - document: 当前输入文档
        - complete_event: 补全事件
        
        返回：
        - 补全建议生成器
        """
        text = document.text_before_cursor
        if document.text_after_cursor.strip():
            return
        # 只有当整个输入以 "/" 开头时才补全
        stripped = text.lstrip()
        if not stripped.startswith("/"):
            return
        # 提取正在输入的标记（最后一个空格之后的部分）
        last_space = text.rfind(" ")
        if last_space >= 0:
            return  # 只补全命令名，不补全参数
        token = stripped[1:]  # 去掉开头的 "/"
        for name in self._names:
            if name.startswith(token):
                yield Completion(
                    text=f"/{name}",
                    start_position=-len(stripped),
                    display=f"/{name}",
                )


# ── 交互式输入会话包装器 ────────────────────────────────────────────


class InputSession:
    """
    基于 prompt_toolkit 的交互式输入会话
    
    相比原生 input() 的优势：
    - 完整的 readline 风格编辑（光标移动、Home/End、删除行等）
    - 会话内的上下箭头历史记录
    - Alt+Enter / Ctrl+J 输入多行，Enter 提交
    - 斜杠命令的 Tab 自动补全

    """

    def __init__(self, command_names: Sequence[str] = ()) -> None:
        """
        初始化输入会话
        
        参数：
        - command_names: 支持的斜杠命令列表
        """
        kb = KeyBindings()

        @kb.add("escape", "enter", eager=True)  # Alt+Enter
        @kb.add("c-j", eager=True)  # Ctrl+J
        def _newline(event: KeyPressEvent) -> None:
            """
            插入换行符（不提交）
            """
            event.current_buffer.insert_text("\n")

        self._session: PromptSession[str] = PromptSession(
            # 提示前缀：显示 "you> "，用亮青色加粗显示
            message=FormattedText([("bold ansibrightcyan", "you> ")]),
            # 多行输入时的续行提示：显示 " ... "，灰色
            prompt_continuation=FormattedText([("ansigray", " ... ")]),
            # 自动补全器：提供斜杠命令的 Tab 补全
            completer=_SlashCompleter(command_names) if command_names else None,
            # 输入时实时补全：开启后边输入边显示补全建议
            complete_while_typing=True,
            # 自定义按键绑定：如 Alt+Enter/Ctrl+J 换行
            key_bindings=kb,
            # 历史记录：使用内存存储本次会话的输入历史（就是类似于命令行的上一次调用，上下箭头调用）
            history=InMemoryHistory(),
            # 多行模式：关闭时 Enter 直接提交，通过快捷键插入换行
            multiline=False,  # Enter 提交；Alt+Enter / Ctrl+J 换行
        )

    def prompt(self) -> str:
        """
        读取用户输入
        
        返回：
        - 用户输入字符串（已去除首尾空白）
        
        可能抛出的异常：
        - EOFError: 用户按下 Ctrl+D
        - KeyboardInterrupt: 用户按下 Ctrl+C
        """
        with patch_stdout(raw=True):
            return self._session.prompt().strip()
