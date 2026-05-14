#!/usr/bin/env python3
"""
LangChain 入门 Demo

这个 demo 展示了 LangChain 的核心功能：
1. 基础聊天模型使用
2. 提示词模板
3. 工具调用 (Agent)
4. 记忆系统
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# 加载环境变量
load_dotenv()


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式
    
    参数:
        expression: 数学表达式字符串，例如 "2 + 2" 或 "10 * 5"
    """
    try:
        # 安全的表达式计算
        import ast
        import operator
        
        # 允许的运算符
        allowed_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
        def eval_expr(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            elif isinstance(node, ast.BinOp):
                op = type(node.op)
                if op not in allowed_ops:
                    raise ValueError(f"不允许的运算符: {op}")
                return allowed_ops[op](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -eval_expr(node.operand)
            else:
                raise ValueError(f"不支持的表达式: {node}")
        
        tree = ast.parse(expression, mode='eval')
        result = eval_expr(tree.body)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前时间: {now}"


def demo_basic_chat():
    """Demo 1: 基础聊天"""
    print("\n" + "="*60)
    print("Demo 1: 基础聊天")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    response = llm.invoke("你好，请用一句话介绍一下 LangChain 是什么")
    print(f"\nAI: {response.content}")


def demo_prompt_template():
    """Demo 2: 提示词模板"""
    print("\n" + "="*60)
    print("Demo 2: 提示词模板")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{job}，请用简洁的语言回答问题。"),
        ("user", "{question}"),
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "job": "Python 导师",
        "question": "什么是 Python 的核心概念？"
    })
    
    print(f"\nAI: {response.content}")


def demo_agent_with_tools():
    """Demo 3: 带工具的 Agent"""
    print("\n" + "="*60)
    print("Demo 3: 带工具的 Agent")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # 定义工具列表
    tools = [calculate, get_current_time]
    
    # 可选：如果有 TAVILY_API_KEY，可以添加搜索工具
    if os.getenv("TAVILY_API_KEY"):
        print("\n检测到 TAVILY_API_KEY，添加搜索工具")
        tools.append(TavilySearchResults(max_results=2))
    
    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手。你可以使用提供的工具来帮助用户回答问题。"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 创建 Agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 测试 Agent
    print("\n问: 计算 100 * 5 + 20")
    result = agent_executor.invoke({
        "input": "计算 100 * 5 + 20"
    })
    print(f"\nAI: {result['output']}")
    
    print("\n问: 现在几点了？")
    result = agent_executor.invoke({
        "input": "现在几点了？"
    })
    print(f"\nAI: {result['output']}")


def demo_conversation_memory():
    """Demo 4: 对话记忆"""
    print("\n" + "="*60)
    print("Demo 4: 对话记忆")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # 创建记忆
    memory = ConversationBufferMemory(return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手，请记住用户说的话。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm
    
    # 第一轮对话
    print("\n用户: 我叫小明，我喜欢编程")
    response = chain.invoke({
        "history": memory.chat_memory.messages,
        "input": "我叫小明，我喜欢编程"
    })
    memory.chat_memory.add_user_message("我叫小明，我喜欢编程")
    memory.chat_memory.add_ai_message(response.content)
    print(f"AI: {response.content}")
    
    # 第二轮对话（测试记忆）
    print("\n用户: 我叫什么名字？我喜欢什么？")
    response = chain.invoke({
        "history": memory.chat_memory.messages,
        "input": "我叫什么名字？我喜欢什么？"
    })
    print(f"AI: {response.content}")


def main():
    """主函数"""
    print("🚀 LangChain 入门 Demo")
    print("="*60)
    
    # 检查 API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  请设置 OPENAI_API_KEY 环境变量！")
        print("可以在项目根目录创建 .env 文件，内容如下：")
        print("OPENAI_API_KEY=你的API密钥")
        return
    
    try:
        demo_basic_chat()
        demo_prompt_template()
        demo_agent_with_tools()
        demo_conversation_memory()
        
        print("\n" + "="*60)
        print("🎉 Demo 完成！")
        print("="*60)
        print("\n✨ LangChain 核心概念总结:")
        print("  • LLMs/ChatModels: 语言模型接口")
        print("  • Prompt Templates: 提示词模板")
        print("  • Chains: 组件链式调用")
        print("  • Agents: 使用工具的智能代理")
        print("  • Memory: 记忆系统")
        print("  • Tools: 工具")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()
