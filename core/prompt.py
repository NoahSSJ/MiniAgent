import json
import os
import platform
from dataclasses import dataclass
import hashlib
from datetime import datetime, timezone
# from tools.workspace import WorkspaceContext

@dataclass
class PromptPrefix():
    prompt_text: str
    prompt_hash: str
    tool_signature: str
    workspace_fingerprint: str
    built_at: str

class PromptManager():
    def __init__(self, model: str, tools: list) -> None:
        self.model = model
        self.tools = tools

    def get_tool_signature(self, tools):
        tools_schema = [t.to_schema() for t in tools]
        return hashlib.sha256(
            json.dumps(tools_schema, sort_keys=True).encode()
        ).hexdigest()
    
    def get_system_prompt(self) -> "PromptPrefix":
        cwd_path = os.getcwd().split('\\')[-1]
        tool_list = "\n".join(f"- **{t.name}**: {t.description}" for t in self.tools)
        uname = platform.uname()
        signature = self.get_tool_signature(self.tools)

        prompt_text = f"""\
            你是 MiniCoder，一个运行在用户终端中的 AI 编程助手。
            你帮助解决软件工程相关的问题：编写代码、修复 bug、重构、解释代码、运行命令等。

            # 环境信息
            - 工作目录：{cwd_path}
            - 操作系统：{uname.system} {uname.release} ({uname.machine})
            - Python 版本：{platform.python_version()}

            # 可用工具
            {tool_list}

            # 加载的模型
            {self.model}

            # 工作规则
            1. **修改前先阅读。** 在修改任何文件之前，务必先阅读该文件的内容。
            2. **小改动用 edit_file。** 针对局部修改使用 edit_file；只有新建文件或完全重写时才使用 write_file。
            3. **验证修改结果。** 做出更改后，运行相关测试或命令来确认修改正确。
            4. **简洁输出。** 多展示代码，少说废话。只在必要时进行解释。
            5. **逐步执行。** 对于多步骤任务，按顺序依次完成。
            6. **edit_file 唯一性匹配。** 使用 edit_file 时，old_string 中要包含足够的上下文，确保能够唯一匹配到目标位置。
            7. **遵循现有风格。** 保持与项目当前编码规范一致。
            8. **不确定时先问。** 如果需求不明确，先向用户提问澄清，不要盲目猜测。
            9. **回复的语言一定要是中文. ** 除非用户指定了某一个语言回复,否则永远都是使用中文回复.
            """
        return PromptPrefix(
            prompt_text=prompt_text,
            prompt_hash=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            workspace_fingerprint='',
            tool_signature=signature,
            built_at=datetime.now(timezone.utc).isoformat()
        )

        