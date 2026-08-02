"""
子agent 委派机制测试。

覆盖:
- DelegateTaskTool 参数校验与回调执行
- 主agent 注册 delegate_task 工具, 子agent(只读)不注册(防递归)
- 子agent 步数受限 / 只读模式 / Session 隔离
- 委派全链路(真实 _delegate): 父调 delegate_task → 子agent读文件 → 纯文本返回父上下文
- 子agent 不持久化 session(不产生垃圾 .session 文件)
- 子agent 长结果裁剪 / 异常兜底
"""
import json
from pathlib import Path
from unittest import mock

import pytest

from core.agent import MiniAgent
from core.session import Role, Session
from core.tools import DelegateTaskTool, ToolExecutionResult


# ---------------------------------------------------------------------------
# 辅助: 可脚本化驱动、且按"角色"路由的 FakeLLM
# ---------------------------------------------------------------------------
def make_fake_llm(parent_script, sub_script):
    """
    按消息内容自动区分父子agent, 各自消费自己的脚本:
    - 消息里含"只读调查员"系统提示 → 子agent脚本
    - 否则 → 父agent脚本

    script 元素:
        {"type": "tool_call", "name": 工具名, "args": dict}
        {"type": "final", "content": "最终回复"}
        {"type": "raise", "error": "抛出的异常信息"}
    """
    def chat(self, messages, tools_schema=[], stream=False):
        is_sub = any(
            m.get("role") == Role.SYSTEM and "只读调查员" in (m.get("content") or "")
            for m in messages
        )
        script = sub_script if is_sub else parent_script
        if not script:
            raise RuntimeError("FakeLLM: 脚本已用尽")
        step = script.pop(0)
        if step["type"] == "raise":
            raise RuntimeError(step["error"])

        class ToolCall:
            def __init__(self, name, arguments):
                self.id = f"call_{name}"
                self.function = mock.Mock()
                self.function.name = name
                self.function.arguments = arguments

        class Msg:
            def __init__(self):
                if step["type"] == "tool_call":
                    self.tool_calls = [ToolCall(step["name"], json.dumps(step["args"]))]
                else:
                    self.tool_calls = None
                self.content = step.get("content", "")

        return Msg()

    return type("FakeLLM", (), {"model": "test-model", "chat": chat})()


# ---------------------------------------------------------------------------
# TestCase 1: DelegateTaskTool
# ---------------------------------------------------------------------------
class TestDelegateTaskTool:
    def test_validate_args_missing_task(self):
        tool = DelegateTaskTool()
        with pytest.raises(ValueError, match="task 参数必填"):
            tool.validate_args()

    def test_validate_args_with_task(self):
        tool = DelegateTaskTool()
        tool.validate_args(task="调查一下")

    def test_execute_without_handler(self):
        tool = DelegateTaskTool()
        msg = tool.execute(task="调查一下")
        assert "未配置处理函数" in msg

    def test_execute_calls_handler(self):
        tool = DelegateTaskTool(handler=lambda task: f"结果: {task}")
        result = tool.execute(task="读 core/agent.py")
        assert result == "结果: 读 core/agent.py"


# ---------------------------------------------------------------------------
# TestCase 2: 工具注册 / 隔离 / 步数 / 只读
# ---------------------------------------------------------------------------
class TestMiniAgentDelegateConfig:
    def setup_method(self):
        llm = type("FakeLLM", (), {"model": "test-model", "chat": lambda *a, **k: None})()
        self.parent = MiniAgent(llm=llm)
        self.sub = MiniAgent(llm=llm, read_only=True, max_steps=3)

    def test_parent_has_delegate_task(self):
        names = [t.name for t in self.parent.tool_manager.tools]
        assert "delegate_task" in names
        schema_names = [s["function"]["name"] for s in self.parent.tool_manager.tools_schema]
        assert "delegate_task" in schema_names

    def test_sub_agent_no_delegate_task(self):
        """只读子agent不应拥有 delegate_task → 无法继续委派, 防止递归"""
        names = [t.name for t in self.sub.tool_manager.tools]
        assert "delegate_task" not in names

    def test_sub_agent_has_limit_steps_and_read_only(self):
        assert self.sub.max_steps == 3
        assert self.sub.read_only is True
        # 父agent默认 50 步, 非只读
        assert self.parent.max_steps == 50
        assert self.parent.read_only is False

    def test_parent_sub_session_isolated(self):
        assert self.parent.session is not self.sub.session

    def test_sub_agent_system_prompt_contains_read_only_rule(self):
        self.sub._inject_sys_prompt()
        sys_msg = self.sub.session.history[0]
        assert sys_msg["role"] == Role.SYSTEM
        assert "只读调查员" in sys_msg["content"]


# ---------------------------------------------------------------------------
# TestCase 3: 委派全链路
# ---------------------------------------------------------------------------
class TestDelegateFlow:
    def test_full_delegate_flow(self, tmp_path):
        """
        真实端到端链路(不 patch _delegate):
        父agent 调 delegate_task → 真实 _delegate 创建只读子agent →
        子agent read_file_tool 读 sample.py → 纯文本结论 → 回到父上下文 → 父最终答复
        """
        target = tmp_path / "sample.py"
        target.write_text("print('hello world')\nprint(42)\n", encoding="utf-8")

        parent_script = [
            {"type": "tool_call", "name": "delegate_task",
             "args": {"task": "读 sample.py 并总结内容"}},
            {"type": "final", "content": "调查完成, 结论如下: sample.py 只有两行 print。"},
        ]
        sub_script = [
            {"type": "tool_call", "name": "read_file_tool",
             "args": {"file_path": str(target)}},
            {"type": "final", "content": "结论: sample.py 只有两行 print 语句。"},
        ]
        llm = make_fake_llm(parent_script, sub_script)

        parent = MiniAgent(llm=llm)
        answer = parent.chat("先调查 sample.py 再总结")

        assert answer == "调查完成, 结论如下: sample.py 只有两行 print。"

        # 父上下文: system → user → assistant(调delegate) → tool(子agent纯文本结果)
        # 注意: 模型最终答复不写回session(原有行为), 所以最后一条是 TOOL
        history = parent.session.history
        roles = [m["role"] for m in history]
        assert roles[:2] == [Role.SYSTEM, Role.USER]
        assert roles[-1] == Role.TOOL
        assert roles[-2] == Role.ASSISTANT

        # TOOL 消息 = 子agent 纯文本结论, 不含 ToolExecutionResult repr
        tool_msg = history[-1]
        assert "结论: sample.py 只有两行 print" in tool_msg["content"]
        assert "ToolExecutionResult" not in tool_msg["content"]

        # 父 session 不包含子agent的只读系统提示(隔离)
        all_text = json.dumps(history, ensure_ascii=False)
        assert "只读调查员" not in all_text

        # 子session 未持久化: 不产生 .session json 文件
        session_dir = Path(__file__).parent.parent / ".session"
        session_files = list(session_dir.glob("*.json")) if session_dir.exists() else []
        assert session_files == []

    def test_result_truncated_to_4000(self):
        """子agent 返回超过 4000 字符的结果会被 _delegate 裁剪"""
        parent_script = [
            {"type": "tool_call", "name": "delegate_task",
             "args": {"task": "调查"}},
            {"type": "final", "content": "done"},
        ]
        # 子agent 一步直接返回 5000 字符长文
        sub_script = [
            {"type": "final", "content": "x" * 5000},
        ]
        llm = make_fake_llm(parent_script, sub_script)
        parent = MiniAgent(llm=llm)

        parent.chat("调查")   # 真实 _delegate 全链路
        tool_msg = parent.session.history[-1]
        assert len(tool_msg["content"]) <= 4000
        assert "结果已截断" in tool_msg["content"]

    def test_delegate_error_returns_error_text(self):
        """子agent 抛异常时, 父agent 收到错误文本而非崩溃"""
        parent_script = [
            {"type": "tool_call", "name": "delegate_task",
             "args": {"task": "调查"}},
            {"type": "final", "content": "ok"},
        ]
        sub_script = [
            {"type": "raise", "error": "boom"},
        ]
        llm = make_fake_llm(parent_script, sub_script)
        parent = MiniAgent(llm=llm)

        parent.chat("调查")
        tool_msg = parent.session.history[-1]
        assert "boom" in tool_msg["content"]

    def test_tool_message_is_plain_text_result(self):
        """普通工具(非委派): TOOL 消息取 content 纯文本, 不含 repr"""
        target = Path("tests/test_delegate.py")
        parent_script = [
            {"type": "tool_call", "name": "read_file_tool",
             "args": {"file_path": str(target), "max_line": 5}},
            {"type": "final", "content": "读完了"},
        ]
        llm = make_fake_llm(parent_script, sub_script=[])
        agent = MiniAgent(llm=llm)
        agent.chat("读一下文件")
        tool_msg = agent.session.history[-1]
        assert tool_msg["role"] == Role.TOOL
        assert tool_msg["content"].startswith("File: ")          # 纯文本文件内容
        assert "ToolExecutionResult(" not in tool_msg["content"] # 无 repr