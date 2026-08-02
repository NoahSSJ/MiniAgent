"""MiniAgent 主程序入口"""
import os
from core.agent import MiniAgent
from dotenv import load_dotenv
from core.llm import MiniLLM
from pathlib import Path

load_dotenv()

llm = MiniLLM(
    model=os.getenv("DS_MODEL", "deepseek-v4-flash"),
    api_key=os.getenv("DS_API_KEY", ""),
    base_url=os.getenv("DS_BASE_URL", "https://api.deepseek.com")
)
agent = MiniAgent(
    llm=llm,
    prompt=None
)

agent.chat("你好,现在是几点.")
# print(Path('.'))
