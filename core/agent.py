from .llm import MiniLLM
from .session import Role, Message, SessionManager
from .logger import logger
from .tools import ToolManager, DelegateTaskTool, ToolExecutionResult
from .prompt import PromptManager
from .context_v2 import ContextManager
from .workspace import WorkspaceContext
import json

class MiniAgent():
    def __init__(
        self,
        llm: MiniLLM,
        prompt: PromptManager = None,
        *,
        max_steps: int = 50,          # 主agent 50步; 子agent传3步
        read_only: bool = False,      # 子agent=True: 只读调查模式
    ) -> None:
        self.llm = llm
        self.session_manager = SessionManager()                       # 每个agent独立Session, 天然隔离
        self.ws = WorkspaceContext(root=r"D:\pico-main\pico").root_path
        self.session = self.session_manager.build()
        # self.session = self.session_manager.load('e6593330-2da5-4807-97b3-96df4f0a0697')
        self.max_steps = max_steps
        self.read_only = read_only
        self.tool_manager = ToolManager(self.session, self.ws)
        # 只有非只读agent才拥有委派工具 → 子agent没有委派工具, 不会递归
        if not read_only:
            self.tool_manager.register(
                DelegateTaskTool(
                    context=self.tool_manager.tool_context,
                    handler=self._delegate,
                )
            )
        self.prompt_manager = PromptManager(self.llm.model, self.tool_manager.tools, self.ws)
        self.context_manager = ContextManager(self.session, self.prompt_manager)
        self._system_prompt_ready = False

    def _delegate(self, task: str) -> str:
        """创建只读子agent去调查任务, 返回纯文本结果"""
        sub_agent = MiniAgent(
            llm=self.llm,
            prompt=None,
            max_steps=3,        # 子agent步数受限(默认3步)
            read_only=True,     # 只读: 不能写文件
        )
        # 给子agent的任务附带只读调查员约束
        research_text = sub_agent.chat(
            f"【只读调查任务】你只有只读工具, 请调查以下内容并输出纯文本结论, "
            f"不要修改任何文件:\n{task}"
        )
        # 裁剪, 防止撑爆父agent上下文(总长严格 ≤ 4000)
        MAX_RESULT_CHARS = 4000
        TRUNC_SUFFIX = "\n...[结果已截断]"
        if len(research_text) > MAX_RESULT_CHARS:
            research_text = research_text[:MAX_RESULT_CHARS - len(TRUNC_SUFFIX)] + TRUNC_SUFFIX
        return research_text

    def _inject_sys_prompt(self):
        if self._system_prompt_ready:
            return 
        system_prompt = self.prompt_manager.get_system_prompt().prompt_text
        # 子agent追加只读约束
        if self.read_only:
            system_prompt += (
                "\n\n# 只读调查员模式\n"
                "你是一个只读调查agent: 只允许查看文件, 禁止任何写入/修改/删除操作。"
                "步数有限, 尽快读完关键文件后直接给出纯文本调查结论。"
            )
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
        # range(50) → range(self.max_steps): 子agent就是3步
        for _ in range(self.max_steps):
            response = self.llm.chat(
                messages=self.session.history.copy(),
                tools_schema=self.tool_manager.tools_schema
            )
            if not response.tool_calls: # type: ignore
                logger.debug(response.content) # type: ignore
                if not one_shot_flag:
                    self.session_manager.save(self.session.session_id)
                return response.content
            else:
                tool_calls_list = []
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
                    # 纯文本: 取 content 字段, 不把 ToolExecutionResult 的 repr 塞进上下文
                    result_text = (
                        result.content
                        if isinstance(result, ToolExecutionResult)
                        else str(result)
                    )
                    self.session.add_message(
                        role=Role.TOOL,
                        content=result_text,
                        tool_calls=None,
                        tool_call_id=tc.id
                    )
        # 步数耗尽兜底
        return "[调查步数已用尽] 已收集信息有限, 请主agent基于已有信息作答。"
                