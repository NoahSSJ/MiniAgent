"""MiniAgent 主程序入口 — 接入模型路由"""
import os
from pathlib import Path

from dotenv import load_dotenv

from core.agent import MiniAgent
from core.llm import MiniLLM
from core.router import ModelRouter, CATEGORY_CHAT, CATEGORY_SIMPLE, CATEGORY_LIFE, CATEGORY_CODE, CATEGORY_CREATION, CATEGORY_COMPLEX

load_dotenv()

# ============ 模型池 ============
# 分类器: 0.5b 最快, 只做输入分类
classifier = MiniLLM(
    model="qwen2.5:0.5b",
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# 轻量模型: 闲聊 / 简单事实 / 生活
llm_small = MiniLLM(
    model="qwen2.5:1.5b",
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# 中等模型: 代码 / 文本创作
llm_medium = MiniLLM(
    model="qwen2.5:7b",
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# 云端强模型(DeepSeek API): 复杂深度推理
llm_strong = MiniLLM(
    model=os.getenv("DS_MODEL", "deepseek-chat"),
    api_key=os.getenv("DS_API_KEY", ""),
    base_url=os.getenv("DS_BASE_URL", "https://api.deepseek.com")
)

# ============ 路由表: 分类 → 模型 ============
route_table = {
    CATEGORY_CHAT: llm_small,       # 闲聊 → 1.5b
    CATEGORY_SIMPLE: llm_small,     # 简单事实 → 1.5b
    CATEGORY_LIFE: llm_small,       # 生活 → 1.5b
    CATEGORY_CODE: llm_medium,      # 代码 → 7b
    CATEGORY_CREATION: llm_medium,  # 文本创作 → 7b
    CATEGORY_COMPLEX: llm_strong,   # 复杂推理 → DeepSeek API
}

router = ModelRouter(
    classifier=classifier,
    route_table=route_table,
    default_model=llm_medium,       # 兜底 7b
)

# ============ Agent ============
# 把路由编排后的模型传给 Agent: 直接用能支撑 tools 的 7b 作为主模型
# (agent 内部需 tools(function calling), 小模型不支持, 故主模型固定用 7b)
agent = MiniAgent(
    llm=llm_medium,
    prompt=None
)

if __name__ == "__main__":

    # Agent 闭环(功能型任务走 7b + tools)
    print("\n--- Agent 任务 ---")
    agent.chat("你好,请介绍一下你自己使用了哪一个模型,并且现在工作区是什么", one_shot_flag=False)