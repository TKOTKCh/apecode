# ApeCode 🦧

一个用 Python 编写的纳米级终端代码代理 —— 一个最小但完整的工具调用 AI 代理实现（类似于 Claude Code / Codex CLI / Kimi CLI），专为学习和实验而构建。

由 [ApeCode.ai](https://apecode.ai) 提供支持

## 功能

- **工具调用代理循环** — `用户 → 模型 → 工具调用 → 工具结果 → 模型 → 响应`，带有可配置的最大步数防护
- **多提供商模型适配器** — OpenAI、Anthropic 和 Kimi（兼容 OpenAI），全部符合统一的 `ChatModel` 协议
- **7 个内置工具** — `list_files`、`read_file`、`write_file`、`replace_in_file`、`grep_files`、`exec_command`、`update_plan`
- **沙箱 + 审批模型** — `SandboxMode`（只读 / 工作区写入 / 危险完全访问）限制路径变更；`ApprovalPolicy`（按需 / 始终 / 从不）控制变更操作的交互式确认
- **插件系统** — 声明式 `apecode_plugin.json` 清单提供工具、斜杠命令和技能
- **MCP 集成** — 通过 `fastmcp` SDK 从 `.mcp.json` / `apecode_mcp.json` 加载外部工具
- **斜杠命令** — `/help`、`/tools`、`/skills`、`/skill`、`/plan`、`/subagents`、`/delegate`、`/exit`
- **子代理委派** — 具有三个默认配置文件的隔离只读代理：`general`、`reviewer`、`researcher`
- **技能模板** — 从 `skills/*/SKILL.md` 目录或插件中发现
- **REPL + 单次执行模式** — 带有 prompt-toolkit 的交互式会话（历史记录、制表符补全、通过 Alt+Enter 多行）或单个提示执行
- **思考模型支持** — 显示思考模型的 `reasoning_content`（例如 Kimi K2.5）
- **AGENTS.md 链** — 从工作区根目录遍历到文件系统根目录，加载 `AGENTS.md` 文件以获取项目特定说明

## 安装

```bash
uv sync
```

依赖项：`openai`、`anthropic`、`fastmcp`、`typer`、`rich`、`prompt-toolkit`。

## 使用方法

### API 密钥

```bash
export OPENAI_API_KEY=your_key       # 用于 provider=openai（默认）
export ANTHROPIC_API_KEY=your_key    # 用于 provider=anthropic
export KIMI_API_KEY=your_key         # 用于 provider=kimi
```

### 交互式 REPL

```bash
uv run ape
```

### 单次执行模式

```bash
uv run ape "read README.md and summarize project structure"
```

### CLI 标志

```bash
uv run ape --provider openai --model gpt-4.1-mini    # 默认
uv run ape --provider anthropic --model claude-sonnet-4-20250514
uv run ape --provider kimi --model kimi-k2.5
uv run ape --max-steps 30 --timeout 180
uv run ape --cwd /path/to/repo
uv run ape --sandbox-mode read-only --approval-policy never
uv run ape --plugin-dir ./plugins
uv run ape --mcp-config ./.mcp.json
uv run ape --skill-dir ./custom-skills
uv run ape --yolo "apply a simple refactor in src/"
uv run ape --version
```

### 斜杠命令（在 REPL 内部）

```
/help                                          — 列出所有命令
/tools                                         — 列出已注册的工具
/skills                                        — 列出已发现的技能
/skill concise-review review src/apecode/cli.py — 使用额外请求运行技能
/plan                                          — 显示当前任务计划
/subagents                                     — 列出子代理配置文件
/delegate reviewer:: review src/apecode/cli.py — 委派给子代理
/exit                                          — 退出
```

## 架构

```
用户输入
    │
    ▼
┌──────────────────────────────────────────────┐
│  cli.py — Typer 应用，_build_runtime，REPL    │
│  ┌────────────────────────────────────────┐  │
│  │  NanoCodeAgent (agent.py)              │  │
│  │  ┌──────────┐    ┌──────────────────┐  │  │
│  │  │ ChatModel │◄──│ model_adapters.py │  │  │
│  │  │ 协议      │   │ OpenAI/Anthropic/ │  │  │
│  │  │           │   │ Kimi 适配器       │  │  │
│  │  └──────────┘    └──────────────────┘  │  │
│  │  ┌──────────────────────────────────┐  │  │
│  │  │ ToolRegistry (tools.py)          │  │  │
│  │  │  7 个内置 + 插件 + MCP 工具      │  │  │
│  │  │  ToolContext：沙箱 + 审批         │  │  │
│  │  └──────────────────────────────────┘  │  │
│  └────────────────────────────────────────┘  │
│  commands.py — 斜杠命令注册                 │
│  plugins.py  — apecode_plugin.json 加载器   │
│  mcp.py      — fastmcp stdio 桥接           │
│  skills.py   — SKILL.md 发现 + 目录         │
│  subagents.py — 隔离的只读委派               │
│  system_prompt.py — 提示构建器 + AGENTS.md  │
│  console.py  — Rich + prompt-toolkit I/O    │
└──────────────────────────────────────────────┘
```

### 模块细分

| 模块 | 用途 |
|---|---|
| `cli.py` | Typer 入口点，组装运行时（`_build_runtime`），运行 REPL 或单次执行 |
| `agent.py` | `NanoCodeAgent` — 使用 `ChatModel` 协议的核心工具调用循环 |
| `tools.py` | `ToolRegistry`、`ToolContext`（沙箱/审批）、7 个内置工具处理程序 |
| `model_adapters.py` | `OpenAIChatCompletionsClient`、`AnthropicMessagesClient`、`KimiChatCompletionsClient` — 所有适配器转换为/从内部 OpenAI 消息格式 |
| `commands.py` | `CommandRegistry` + `SlashCommand` — `/help`、`/tools`、`/exit` 等 |
| `plugins.py` | 加载 `apecode_plugin.json` 清单；注册工具、命令、技能 |
| `mcp.py` | 解析 `.mcp.json`，通过 `fastmcp.Client` 连接，注册 MCP 工具 |
| `skills.py` | `SkillCatalog` — 发现 `SKILL.md` 文件，支持插件提供的技能 |
| `subagents.py` | `SubagentRunner` — 生成具有只读工具和限制步数的隔离代理 |
| `system_prompt.py` | 使用环境信息、AGENTS.md 链、技能目录构建系统提示 |
| `console.py` | Rich 控制台输出（面板、微调器、工具调用显示）+ prompt-toolkit 输入会话 |

## 环境变量

| 变量 | 默认值 | 描述 |
|---|---|---|
| `APECODE_PROVIDER` | `openai` | 模型提供商（`openai` / `anthropic` / `kimi`） |
| `APECODE_MODEL` | `gpt-4.1-mini` | 模型名称 |
| `APECODE_SANDBOX_MODE` | `workspace-write` | 沙箱模式（`read-only` / `workspace-write` / `danger-full-access`） |
| `APECODE_APPROVAL_POLICY` | `on-request` | 审批策略（`on-request` / `always` / `never`） |
| `OPENAI_API_KEY` | — | OpenAI API 密钥 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 自定义 OpenAI 兼容端点 |
| `ANTHROPIC_API_KEY` | — | Anthropic API 密钥 |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com/v1` | 自定义 Anthropic 端点 |
| `ANTHROPIC_API_VERSION` | `2023-06-01` | Anthropic API 版本头 |
| `KIMI_API_KEY` | — | Kimi API 密钥 |
| `KIMI_BASE_URL` | `https://api.moonshot.cn/v1` | Kimi 端点 |

## 插件系统

将插件清单作为 `apecode_plugin.json` 放置在插件目录中：

```json
{
  "name": "EchoPlugin",
  "tools": [
    {
      "name": "echo_text",
      "description": "从 JSON 参数中回显文本",
      "parameters": {
        "type": "object",
        "properties": { "text": { "type": "string" } },
        "required": ["text"],
        "additionalProperties": false
      },
      "argv": ["python3", "/absolute/path/to/tool.py"],
      "mutating": false,
      "timeout_sec": 60
    }
  ],
  "commands": [
    {
      "name": "quick-review",
      "description": "运行插件提示模板",
      "usage": "/quick-review <task>",
      "output": "正在运行快速审查...",
      "agent_input_template": "审查此任务：\\n{args}"
    }
  ],
  "skills": [
    {
      "name": "plugin-skill",
      "description": "一个插件提供的技能",
      "content": "# 插件技能\\n\\n保持输出简洁。"
    }
  ]
}
```

- 工具使用 `argv`（推荐）或 `command` 来指定可执行文件。
- 工具进程在 `stdin` 上接收 JSON 参数，并将结果写入 `stdout`。
- 命令在 `agent_input_template` 中支持 `{args}` 占位符。
- 技能可以使用内联 `content` 或相对于清单的 `file` 路径。

## MCP 配置

从工作区根目录中的 `.mcp.json` 或 `apecode_mcp.json` 加载 MCP 工具，或通过 `--mcp-config`：

```json
{
  "mcpServers": {
    "demo": {
      "command": "python3",
      "args": ["/absolute/path/to/mcp_server.py"],
      "timeout_sec": 30
    }
  }
}
```

## 技能

在 `skills/<name>/SKILL.md` 中创建一个技能：

```markdown
# concise-review

审查代码并用简洁的项目符号回答。
```

在 REPL 内部使用：

```
/skill concise-review review src/apecode/agent.py
```

## 开发

```bash
# 安装开发依赖
uv sync

# 运行测试
uv run pytest

# 运行单个测试文件
uv run pytest tests/test_tools.py -v

# 检查代码
uv run ruff check src/ tests/

# 自动修复代码
uv run ruff check --fix src/ tests/

# 格式化代码
uv run ruff format src/ tests/
```

## 项目结构

```
src/apecode/
├── __init__.py          # 包版本
├── __main__.py          # python -m apecode 入口
├── cli.py               # Typer CLI 应用 + 运行时组装
├── agent.py             # NanoCodeAgent 核心循环
├── tools.py             # 工具注册 + 内置工具
├── model_adapters.py    # 模型适配器（OpenAI/Anthropic/Kimi）
├── commands.py          # 斜杠命令框架
├── plugins.py           # 插件清单加载器
├── mcp.py               # MCP stdio 桥接
├── skills.py            # 技能发现 + 目录
├── subagents.py         # 子代理委派
├── system_prompt.py     # 系统提示构建器
└── console.py           # Rich + prompt-toolkit I/O
tests/
├── test_agent.py
├── test_commands.py
├── test_mcp.py
├── test_model_adapters.py
├── test_plugins.py
├── test_skills.py
├── test_subagents.py
└── test_tools.py
```

## 许可证

Apache-2.0
