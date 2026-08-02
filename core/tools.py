from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import concurrent
import json
import os
from pathlib import Path
from typing import Optional
from .session import Role, Session
from .logger import logger

class BaseTool(ABC):
    name: str
    description: str
    parameters: dict
    is_risk: bool

    def __init__(self, context = None):
        super().__init__()
        self.context = context

    @abstractmethod
    def validate_args(self, **kwargs):
        ...

    @abstractmethod
    def execute(self, **kwargs):
        ...

    def to_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
class GetCurrentTime(BaseTool):
    name = "get_current_time"
    description = "get_current_time"
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    is_risk = False

    def validate_args(self):
        pass

    def execute(self):
        return datetime.now().strftime("%Y%m%d-%H%M%S")
    
class ReadFileTool(BaseTool):
    name = "read_file_tool"
    description = "阅读文件的工具:阅读一个带着行号的文件的内容,在编辑文件之前总是先阅读文件."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "需要阅读的文件的路径"
            },
            "start_line": {
                "type": "integer",
                "description": "开始的行号,默认是从第1行开始"
            },
            "max_line": {
                "type": "integer",
                "description": "最多读取的行数（从start_line开始算起），默认2000行"
            }
        },
        "required": ["file_path"]
    }
    is_risk = False

    def validate_args(self, **kwargs):
        logger.debug(f" >>> validate ReadFileTool args")
        file_path = kwargs.get("file_path")
        # 检查文件参数变量是否存在
        if not file_path:
            raise ValueError(f"Error: file_path not found")
        # 检查是否存在路径越界问题
        if self.context:
            try:
                abs_path = self.context.path(file_path)
            except PermissionError as e:
                raise ValueError(f"路径越界校验失败: {e}")
        else:
            abs_path = Path(file_path).resolve()
            logger.debug(f"  >>> [validate_tool]   路径解析为: {abs_path}")

        # 检查文件路径是否存在
        if not abs_path.exists():
            raise ValueError(f"参数校验失败: 文件 '{file_path}' 不存在!")
        if not abs_path.is_file():
            raise ValueError(f"参数校验失败: '{file_path}' 不是一个文件")
        
    
    def execute(self, file_path: str, start_line: int = 1, max_line: int = 2000):
        p = Path(file_path).expanduser().resolve()
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        start = max(0, start_line - 1)
        chunk = lines[start: (start + max_line)]
        result = []
        header = f"File: {file_path}\nTotal lines: {total}"
        result.append(header)
        for i, line in enumerate(chunk, start=start + 1):
            result.append(f"{i} | {line}")
        print("\n".join(result))
        return "\n".join(result)
        
class DelegateTaskTool(BaseTool):
    """委派调查工具: 把子任务交给只读子agent, 执行逻辑由handler回调注入"""
    name = "delegate_task"
    description = (
        "当任务需要先做调查（如阅读多个文件、梳理代码结构、评估影响）时，"
        "把该子任务委派给一个只读子agent去调查，返回纯文本调查结果。"
        "在动手修改或下结论之前，信息不足优先调用本工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "需要交给子agent调查的任务描述"}
        },
        "required": ["task"]
    }
    is_risk = False

    def __init__(self, context=None, handler=None):
        super().__init__(context)
        self.handler = handler          # handler(task: str) -> str

    def validate_args(self, **kwargs):
        if not kwargs.get("task"):
            raise ValueError("task 参数必填")

    def execute(self, task: str, **kwargs):
        if self.handler is None:
            return "Error: delegate_task 未配置处理函数"
        return self.handler(task)


@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def build(
        cls,
        content: str,
        tool_status: str,
        tool_error_code: str = "",
        security_event_type: str = "",
        risk_level: str = "low",
        read_only: bool = True,
        affected_paths: Optional[list[str]] = None,
        workspace_changed: bool = False,
        workspace_fingerprint: str = "",
        diff_summary: Optional[list[str]] = None,
    ) -> "ToolExecutionResult":
        metadata = {
            "tool_status": tool_status,
            "tool_error_code": tool_error_code,
            "security_event_type": security_event_type,
            "risk_level": risk_level,
            "read_only": read_only,
            "affected_paths": affected_paths or [],
            "workspace_changed": workspace_changed,
            "diff_summary": diff_summary or [],
        }
        if workspace_fingerprint:
            metadata["workspace_fingerprint"] = workspace_fingerprint
        return cls(content=content, metadata=metadata)
    
class ToolContext():
    """
    只负责路径越界安全检测,路径存不存在并不关心
    """
    def __init__(
            self,
            root: Path,
            depth: int = 0,
            max_depth: int = 1,
    ) -> None:
        self.root = root.resolve()
        self.depth = depth
        self.max_depth = max_depth

    def path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self.root / p).resolve()
        root_str = os.path.normcase(str(self.root))
        resolved_str = os.path.normcase(str(resolved))
        if not resolved_str.startswith(root_str + os.sep) and resolved_str != root_str:
            raise PermissionError(
                f"路径越界! '{resolved}'不在工作区{self.root}内"
            )
        return resolved

class ToolManager():
    """
    ToolManager:管理工具的注册/执行/校验
    """
    def __init__(self, session: Session) -> None:
        self.tool_context = ToolContext(
            root=Path('.'),
            depth=0,
            max_depth=3,
        )
        self.session = session
        self.tools = [] 
        self.tools_name = {}
        self.tools_schema = []
        self.white_list = []
        self.create_default_register()

    def register(self, tool):
        self.tools.append(tool)
        self.tools_name[tool.name] = tool
        self.tools_schema.append(tool.to_schema())
        self.white_list.append(tool.name)
        logger.debug(f"tool: {tool.name} register success.")

    def create_default_register(self):
        default_tools = [
            GetCurrentTime,
            ReadFileTool
        ]
        for tool_class in default_tools:
            if self.tool_context is not None:
                tool_instance = tool_class(context=self.tool_context)
            else:
                tool_instance = tool_class()
            self.register(tool_instance)

    def execute(self, name: str, args: dict) -> ToolExecutionResult:
        # 第一层： 白名单检查，工具是否允许
        logger.debug(f" >>> tool {name} whitelist checking")
        if self.white_list is not None and self.white_list != [] and name not in self.white_list:
            return ToolExecutionResult.build(
                content=f"Error: tool '{name}' is not in tools white list.",
                tool_status="reject",
                tool_error_code="tool_not_allowed",
                risk_level="high",
                read_only=False
            )
        logger.debug(f" >>> tool {name} whitelist checked success")
        # 第二层：工具是否存在
        logger.debug(f" >>> tool {name} exist checking")
        tool_instance = self.tools_name.get(name)
        if tool_instance is None:
            return ToolExecutionResult.build(
                content=f"Error: unknown tool '{name}'.",
                tool_status="reject",
                tool_error_code="unknown_tool",
                read_only=False
            )
        logger.debug(f" >>> tool {name} exist checked success.")

        # 第三层：工具函数参数校验
        logger.debug(f" >>> tool {name} validating args.")
        try:
            tool_instance.validate_args(**args)
        except Exception as e:
            return ToolExecutionResult.build(
                content=f"Error: Validation failed {e}",
                tool_status="reject",
                tool_error_code="Validation_failed",
                risk_level="low",
                read_only=True
            )
        logger.debug(f" >>> tool {name} validate args success.")
        
        # 第四层： 重复调用检测
        logger.debug(f" >>> tool {name} repeat calls checking.")
        if self.is_repeat_tool_call(name=name, args=args):
            return ToolExecutionResult.create(
                content=f"error: repeated identical tool call for {name}; choose a different tool or return a final answer",
                tool_status="rejected",
                tool_error_code="repeated_identical_call",
                # risk_level="high" if self._is_risky(name) else "low",
                read_only=False
            )
        logger.debug(f" >>> tool {name} repeat calls not found.")

        # 第五层： 危险工具审批

        # 执行前快照

        # 执行工具
        logger.debug(f" >>> tool {name} executing.")
        try:
            result = tool_instance.execute(**args)
        except Exception as e:
            return ToolExecutionResult.build(
                content=f"Error: tool {name} act failed {e}.",
                tool_status="error",
                tool_error_code="tool_failed",
                risk_level="high" if tool_instance.is_risk else "low",
                read_only=False
            )
        logger.debug(f" >>> tool {name} execute success.")

        # 执行后快照

        # 输出裁剪
        return ToolExecutionResult.build(
            content=result,
            tool_status="ok",
            risk_level="high" if tool_instance.is_risk else "low",
            read_only=True
        )
    
    def _execute_tool(self, name: str, args: dict) -> ToolExecutionResult:
        return self.execute(name, args)
    
    def _execute_tools(self, tool_list: list[tuple[str, dict]]) -> list[ToolExecutionResult]:
        """并行执行多个工具"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(self.execute, name, args)
                for name, args in tool_list
            ]
            return [f.result() for f in concurrent.futures.as_completed(futures)]
    
    def is_repeat_tool_call(self, name: str, args: dict) -> bool:
        if self.session is None:
            return False
        history = self.session.get_history()
        prev_calls = []
        for msg in history:
            if msg.get("role") == Role.ASSISTANT and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"]["arguments"])
                    prev_calls.append({"name": tool_name, "args": tool_args})
        
        if len(prev_calls) < 2:
            return False
        
        recent = prev_calls[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)
    
                    



        
