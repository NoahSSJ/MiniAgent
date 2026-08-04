from openai import OpenAI
from pprint import pprint
import logging
import httpx
from .logger import logger
from .session import Message
from typing import Union

class MiniLLM():
    def __init__(self, model: str, api_key: str, base_url: str, **kwargs) -> None:
        self.model = model
        # trust_env=False: 禁用系统代理, 避免 localhost 请求被 Clash 等代理劫持(导致 502)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(trust_env=False)
        )
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_costs = 0
        self.max_tokens = kwargs.get("max_tokens")
        self.max_context_tokens = kwargs.get("max_context_tokens")
        self.temperture = kwargs.get("temperture")

    def chat(self, messages: list[dict] = [], tools_schema: list = [], stream: bool = False, response_format: dict = None):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": stream
        }
        if tools_schema:
            kwargs["tools"] = tools_schema
        if response_format:
            kwargs["response_format"] = response_format
        response = self.client.chat.completions.create(**kwargs)
        if stream:
            pass
        else:
            msg = response.choices[0].message
            logger.debug(msg)
            return msg