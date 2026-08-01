from .llm import MiniLLM
from .session import Role, Message, Session
from .logger import logger
from .tools import ToolManager
from .prompt import PromptManager
from pprint import pprint
import json

class MiniAgent():
    def __init__(self, llm: MiniLLM, prompt: PromptManager) -> None:
        self.llm = llm
        self.session = Session()
        self.tool_manager = ToolManager(self.session)
        self.prompt = PromptManager(self.llm.model, self.tool_manager.tools)
        self._system_prompt_ready = False

    def _inject_sys_prompt(self):
        if self._system_prompt_ready:
            return 
        system_prompt = self.prompt.get_system_prompt().prompt_text
        self.session.add_message(
            role = Role.SYSTEM,
            content = system_prompt,
            tool_calls=None,
            tool_call_id=None
        )
        self._system_prompt_ready = True

        

    def chat(self, user_input: str, one_shot_flag: bool = True):
        self._inject_sys_prompt()
        self.session.add_message(
            role=Role.USER,
            content=user_input,
            tool_calls=None,
            tool_call_id=None
        )
        for _ in range(50):
            response = self.llm.chat(
                messages=self.session.history.copy(),
                tools_schema=self.tool_manager.tools_schema
            )
            if not response.tool_calls: # type: ignore
                logger.debug(response.content) # type: ignore
                if not one_shot_flag:
                    self.session.save()
                return response.content
            else:
                tool_calls_list = []
                print(response.tool_calls)
                for tc in response.tool_calls:
                    tool_calls_list.append({
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                        "type": "function",
                    })
                self.session.add_message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=tool_calls_list,
                    tool_call_id=None
                )
                for tc in response.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)
                    result = self.tool_manager._execute_tool(tool_name, tool_args)
                    self.session.add_message(
                        role=Role.TOOL,
                        content=str(result),
                        tool_calls=None,
                        tool_call_id=tc.id
                    )
                