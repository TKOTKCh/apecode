"""
技能发现和渲染模块

核心功能：
- 从文件系统发现 SKILL.md 技能文件
- 解析技能内容和描述
- 构建技能目录索引
- 为系统提示词格式化技能列表

技能目录结构示例：
project/
├── SKILL.md              # 根目录技能
├── code-review/
│   └── SKILL.md          # 子目录技能
└── .system/              # 被忽略的系统目录
    └── SKILL.md
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# 技能文件的固定名称
SKILL_FILE_NAME = "SKILL.md"


@dataclass(frozen=True, slots=True)
class Skill:
    """
    单个技能定义
    
    技能可以来自：
    1. 文件系统（path 有值）
    2. 内联内容（inline_content 有值，如插件中的技能）
    
    字段：
    - name: 技能名称（标准化后的）
    - description: 技能描述（从内容提取）
    - path: 技能文件路径（如果来自文件）
    - inline_content: 内联技能内容（如果来自内联）
    - source: 技能来源标识
    """

    name: str
    description: str
    path: Path | None = None
    inline_content: str | None = None
    source: str = "local"

    def read_text(self) -> str:
        """
        读取技能内容文本
        
        优先级：
        1. inline_content（内联内容）
        2. path（文件内容）
        3. 空字符串
        
        返回：
        - 技能内容字符串（去除首尾空白）
        """
        if self.inline_content is not None:
            return self.inline_content.strip()
        if self.path is None:
            return ""
        return self.path.read_text(encoding="utf-8", errors="replace").strip()


def _extract_description(text: str) -> str:
    """
    从技能内容中提取描述
    
    提取规则：
    1. 跳过空行
    2. 跳过以 # 开头的标题行
    3. 取第一个非空非标题行的前 160 个字符
    4. 如果没有符合条件的行，返回 "No description."
    
    参数：
    - text: 技能内容文本
    
    返回：
    - 技能描述字符串
    """
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        return line[:160]
    return "No description."


def _iter_skill_files(roots: Iterable[Path]) -> list[Path]:
    """
    迭代查找所有技能文件
    
    查找逻辑：
    1. 对每个根目录，先检查目录本身是否有 SKILL.md
    2. 再检查直接子目录是否有 SKILL.md（一层嵌套）
    
    参数：
    - roots: 要搜索的根目录列表
    
    返回：
    - 找到的所有技能文件路径列表
    """
    discovered: list[Path] = []
    for root in roots:
        normalized = root.expanduser().resolve()
        if not normalized.exists() or not normalized.is_dir():
            continue

        # 检查根目录本身是否有技能文件
        direct = normalized / SKILL_FILE_NAME
        if direct.exists() and direct.is_file():
            discovered.append(direct)

        # 检查直接子目录是否有技能文件
        for child in sorted(normalized.iterdir()):
            if not child.is_dir():
                continue
            nested = child / SKILL_FILE_NAME
            if nested.exists() and nested.is_file():
                discovered.append(nested)
    return discovered


@dataclass(slots=True)
class SkillCatalog:
    """
    内存中的技能目录索引
    
    功能：
    - 从文件系统构建技能索引
    - 按名称查找技能
    - 格式化技能列表输出
    - 合并额外的技能
    
    字段：
    - _skills: 技能字典，key 是标准化后的技能名
    """

    _skills: dict[str, Skill]

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        标准化技能名称
        
        处理步骤：
        1. 去除首尾空白
        2. 转小写
        3. 空格替换为连字符
        
        示例：
        "Code Review" → "code-review"
        "  MySkill  " → "myskill"
        
        参数：
        - name: 原始技能名称
        
        返回：
        - 标准化后的技能名称
        """
        return name.strip().lower().replace(" ", "-")

    @classmethod
    def from_roots(cls, roots: Iterable[Path]) -> SkillCatalog:
        """
        从根目录列表构建技能目录
        
        执行流程：
        1. 查找所有 SKILL.md 文件
        2. 跳过 .system 目录中的技能
        3. 用父目录名作为技能名
        4. 读取文件内容并提取描述
        5. 构建索引字典
        
        参数：
        - roots: 要搜索的根目录列表
        
        返回：
        - 构建好的 SkillCatalog 对象
        """
        indexed: dict[str, Skill] = {}
        for path in _iter_skill_files(roots):
            # 跳过 .system 系统目录
            if path.parent.name.lower() == ".system":
                continue
            # 用父目录名作为技能名
            name = cls._normalize_name(path.parent.name)
            # 避免重复
            if name in indexed:
                continue
            # 读取文件内容
            content = path.read_text(encoding="utf-8", errors="replace")
            # 创建技能对象
            indexed[name] = Skill(
                name=name,
                description=_extract_description(content),
                path=path,
                # 只提取description和路径
                # 主要为了实现渐进式加载，这里不直接把所有内容content赋值给inline_content，而是用None表示从文件读取？
                source=f"file:{path}",
            )
        return cls(_skills=indexed)

    def with_additional(self, skills: Iterable[Skill]) -> SkillCatalog:
        """
        返回一个合并了额外技能的新目录（不可变操作）
        
        合并规则：
        1. 保留原目录的所有技能
        2. 添加新技能（按名称去重）
        3. 跳过已存在的技能名
        4. 返回新的 SkillCatalog 对象
        
        参数：
        - skills: 要添加的技能列表
        
        返回：
        - 新的 SkillCatalog 对象
        """
        indexed = dict(self._skills)
        for skill in skills:
            normalized = self._normalize_name(skill.name)
            if not normalized or normalized in indexed:
                continue
            indexed[normalized] = Skill(
                name=normalized,
                description=skill.description,
                path=skill.path,
                inline_content=skill.inline_content,
                source=skill.source,
            )
        return SkillCatalog(_skills=indexed)

    def list_skills(self) -> list[Skill]:
        """
        返回所有技能列表（按名称排序）
        
        返回：
        - 排序后的技能列表
        """
        return [self._skills[name] for name in sorted(self._skills)]

    def get(self, name: str) -> Skill | None:
        """
        通过技能名称查找技能（不区分大小写）
        
        参数：
        - name: 技能名称
        
        返回：
        - 找到的 Skill 对象，没找到返回 None
        """
        return self._skills.get(self._normalize_name(name))

    def format_overview(self) -> str:
        """
        渲染简洁的技能概览（用于快速展示）
        
        格式示例：
        - code-review: 代码审查指南
        - test-writing: 如何编写测试
        
        返回：
        - 格式化的概览字符串
        """
        skills = self.list_skills()
        if not skills:
            return "(none)"
        return "\n".join(f"- {skill.name}: {skill.description}" for skill in skills)

    def format_for_system_prompt(self) -> str:
        """
        渲染用于系统提示词的技能说明，用于初始化系统提示词
        
        包含内容：
        1. 技能概念说明
        2. 可用技能列表（带来源）
        3. 技能使用指南
        
        返回：
        - 格式化的系统提示词字符串
        """
        skills = self.list_skills()
        if not skills:
            return "(none)"
        lines = [
            "A skill is a local instruction bundle stored in `SKILL.md`.",
            "### Available skills",
        ]
        # 装载可用skill列表，这里只加来源和description
        for skill in skills:
            location = str(skill.path) if skill.path is not None else skill.source
            lines.append(f"- {skill.name}: {skill.description} (source: {location})")
        lines.extend(
            [
                "### How to use skills",
                "- If the user names a skill explicitly, use it in this turn.",
                "- Read only the needed part of `SKILL.md` to keep context small.",
                "- Resolve relative references from the skill directory first.",
                "- If a skill cannot be loaded, explain briefly and fallback.",
            ]
        )
        return "\n".join(lines)
