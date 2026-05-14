# LangChain Demo

这个目录包含了 LangChain 的入门示例代码。

## 安装依赖

首先，你需要安装 LangChain 相关依赖：

```bash
# 使用 uv（推荐）
uv pip install langchain langchain-openai langchain-community python-dotenv

# 或者使用 pip
pip install langchain langchain-openai langchain-community python-dotenv
```

## 配置环境变量

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=你的OpenAI密钥
TAVILY_API_KEY=你的Tavily密钥（可选，用于搜索）
```

## 运行 Demo

```bash
cd examples
python langchain_demo.py
```

## Demo 内容

这个 demo 包含 4 个示例：

1. **基础聊天** - 直接与 LLM 对话
2. **提示词模板** - 使用模板动态生成提示词
3. **带工具的 Agent** - 展示如何让 AI 使用工具（计算、获取时间、搜索）
4. **对话记忆** - 保持上下文的对话

## LangChain 核心概念

| 概念 | 说明 |
|------|------|
| LLMs/ChatModels | 语言模型接口 |
| Prompt Templates | 提示词模板 |
| Chains | 组件链式调用 |
| Agents | 使用工具的智能代理 |
| Memory | 记忆系统 |
| Tools | 工具 |

## 与 ApeCode 的对比

| 特性 | ApeCode | LangChain |
|------|---------|-----------|
| 复杂度 | 简单，几百行代码 | 复杂，数万行代码 |
| 抽象层级 | 低 | 高 |
| 适用场景 | 专门的代码助手 | 通用 LLM 应用框架 |
| 学习曲线 | 很低 | 较高 |
