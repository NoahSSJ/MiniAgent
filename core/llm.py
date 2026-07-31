from openai import OpenAI
from pprint import pprint
import logging
from .logger import logger

class MiniLLM():
    def __init__(self, model: str, api_key: str, base_url: str, **kwargs) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_costs = 0
        self.max_tokens = kwargs.get("max_tokens")
        self.max_context_tokens = kwargs.get("max_context_tokens")
        self.temperture = kwargs.get("temperture")

    def chat(self, messages: list[dict], tools_schema: dict, stream: bool = False) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": stream
        }
        if tools_schema:
            kwargs["tools"] = tools_schema
        response = self.client.chat.completions.create(**kwargs)
        if stream:
            return ''
        else:
            msg = response.content
            logger.debug(msg)
            return msg
        