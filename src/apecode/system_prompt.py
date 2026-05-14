"""
系统提示词构建模块

核心功能：
- 从当前目录向上查找 AGENTS.md 文件
- 构建功能完整的默认系统提示词
- 集成环境信息、技能列表、AGENTS.md 等内容

AGENTS.md 查找逻辑：
- 从当前目录开始，逐级向上到文件系统根目录
- 同时支持 AGENTS.md 和 agents.md（大小写兼容）
- 结果按从根目录到当前目录的顺序排列（根目录在前）
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def find_agents_md(cwd: Path) -> list[Path]:
    """
    从当前目录向上查找 AGENTS.md 文件
    
    查找逻辑：
    1. 从当前目录开始
    2. 检查目录中是否有 AGENTS.md 或 agents.md
    3. 向上移动到父目录，重复步骤 2
    4. 直到到达文件系统根目录（parent == self）
    5. 反转结果，使根目录的文件排在前面
    
    参数：
    - cwd: 当前工作目录路径
    
    返回：
    - 找到的 AGENTS.md 文件路径列表（从根目录到当前目录排序）
    
    示例目录结构：
    /project/AGENTS.md      ← 会被找到
    /project/subdir/AGENTS.md  ← 会被找到
    /other/AGENTS.md        ← 不会被找到（不在路径上）
    """
    results: list[Path] = []
    current = cwd.resolve()
    while True:
        # 同时检查大写和小写的文件名
        for filename in ("AGENTS.md", "agents.md"):
            candidate = current / filename
            if candidate.exists() and candidate.is_file():
                results.append(candidate)
        # 到达文件系统根目录时停止（parent 就是自己）
        if current.parent == current:
            break
        current = current.parent
    # 反转结果，使根目录的文件排在前面
    results.reverse()
    return results


def build_system_prompt(cwd: Path, *, skills_overview: str | None = None, dir_listing: str | None = None) -> str:
    """
    构建功能完整的默认系统提示词（包含环境提示）
    
    提示词包含内容：
    1. ApeCode 身份定义
    2. 核心原则（简洁、验证、最小改动等）
    3. 工具使用策略（优先专用工具，避免 exec_command 滥用）
    4. 编码指南（现有代码、新代码）
    5. Git 安全规则
    6. 研究探索策略
    7. 工作环境信息（时间、目录、文件列表）
    8. AGENTS.md 自定义指令（优先级高于默认）
    9. 技能列表
    10. 提醒事项
    
    参数：
    - cwd: 工作区根目录路径
    - skills_overview: 技能概览字符串（来自 SkillCatalog）
    - dir_listing: 顶层目录列表字符串（可选）
    
    返回：
    - 完整的系统提示词字符串
    
    注意：
    - AGENTS.md 的指令优先级高于本函数中的默认值
    - 时间使用 UTC 时区
    """
    now = datetime.now(UTC).isoformat()
    agents_blocks: list[str] = []
    # 读取所有找到的 AGENTS.md 文件
    for file in find_agents_md(cwd):
        content = file.read_text(encoding="utf-8", errors="replace").strip()
        agents_blocks.append(f"## {file}\n{content}")
    agents_text = "\n\n".join(agents_blocks) if agents_blocks else "(none)"

    # 构建技能列表部分
    skills_text = skills_overview.strip() if skills_overview else "(none)"

    # 构建目录列表部分（如果提供）
    dir_listing_section = ""
    if dir_listing:
        dir_listing_section = f"- Top-level directory listing:\n{dir_listing}\n"

    return (
        "You are ApeCode, a terminal coding agent.\n"
        "You collaborate with the user to complete coding and research tasks safely and efficiently.\n\n"
        "# Core Principles\n"
        "- Be concise, direct, and helpful.\n"
        "- Respond in the same language as the user unless asked otherwise.\n"
        "- Think step-by-step for complex tasks. Break down problems before acting.\n"
        "- Verify before assuming — use tools to check facts rather than guessing.\n"
        "- Do not hallucinate file paths, function names, or APIs. If unsure, search first.\n"
        "- Keep changes minimal and focused on the requested goal.\n\n"
        "# Tool Usage Strategy\n"
        "Always prefer dedicated tools over exec_command for common operations:\n"
        "- Directory listing → list_files (not ls or find)\n"
        "- Reading files → read_file (not cat, head, or tail)\n"
        "- Searching file contents → grep_files (not grep or rg)\n"
        "- Editing existing files → replace_in_file (not sed or awk)\n"
        "- Creating new files → write_file\n\n"
        "Use exec_command only for: running tests, git operations, build commands, package management, "
        "and other system tasks that have no dedicated tool.\n\n"
        "Read before write: always read_file before using replace_in_file so you know the exact text to match.\n\n"
        "If multiple independent reads or searches are needed, issue them as parallel tool calls in one response.\n\n"
        "For tasks with 3+ steps, use update_plan to track progress and keep the user informed.\n\n"
        "For mutating actions, follow runtime approval and sandbox policies.\n\n"
        "# Coding Guidelines\n"
        "## Working with Existing Code\n"
        "- Read and understand the relevant code before making changes.\n"
        "- Follow the existing project style, conventions, and structure.\n"
        "- Prefer root-cause fixes over superficial patches.\n"
        "- Make minimal, focused changes — avoid unrelated refactors unless explicitly asked.\n\n"
        "## Writing New Code\n"
        "- Match the project's coding style (naming, formatting, patterns).\n"
        "- Add tests when it is natural and expected in this codebase.\n"
        "- Avoid introducing unnecessary dependencies or abstractions.\n\n"
        "# Git Safety\n"
        "- Never force-push or amend published commits without explicit user approval.\n"
        "- Do not commit files that may contain secrets (.env, credentials, API keys).\n"
        "- Do not push to main/master without user confirmation.\n"
        "- Prefer creating new commits over amending existing ones.\n\n"
        "# Research and Exploration\n"
        "- Start exploration with a non-recursive list_files to understand project layout.\n"
        "- Use grep_files to trace function calls, imports, and references across the codebase.\n"
        "- State assumptions explicitly and verify them with tools before acting.\n\n"
        "# Working Environment\n"
        f"- Current UTC time: {now}\n"
        f"- Workspace root: {cwd}\n"
        f"{dir_listing_section}\n"
        "# AGENTS.md Instructions\n"
        "AGENTS.md instructions take precedence over the defaults above when they conflict.\n\n"
        f"{agents_text}\n\n"
        "# Skills\n"
        f"{skills_text}\n\n"
        "# Reminders\n"
        "- Be helpful, thorough, and patient.\n"
        "- When errors occur, diagnose the root cause rather than retrying blindly.\n"
        "- Think twice before irreversible changes — confirm with the user if unsure.\n"
        "- Keep it simple. The best solution is the simplest one that works correctly.\n"
    )
