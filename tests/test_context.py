"""
core.context 上下文预算管理器测试（Mini 版）。

覆盖:
- 正常组装: 不超预算, 不触发降级, system + user 均在
- 三级渐进式降级: 超预算时 L1工具摘要 → L2 FIFO丢历史 → L3 强制截断
- 驻留区保护: system 与 user 在任何降级下都绝不参与裁剪
- 边界: 小预算下 L3 兜底不崩溃, 预算充足时全量渲染
"""
from core.context_v1 import ContextManager, Section


# ---------------------------------------------------------------------------
# 辅助: 假 agent / prompt / session
# ---------------------------------------------------------------------------
class FakePrompt:
    def __init__(self, text="SYSTEM-指令-永不裁剪-"):
        self.text = text

    def get_system_prompt(self):
        class P:
            pass
        p = P()
        p.prompt_text = self.text
        return p


class FakeSession:
    def __init__(self, history):
        self.history = history


def make_agent(system_text, history):
    return type("Agent", (), {"prompt": FakePrompt(system_text), "session": FakeSession(history)})()


# ---------------------------------------------------------------------------
# 正常组装: 不超预算 → 不降级
# ---------------------------------------------------------------------------
class TestNormalAssembly:
    def test_no_degradation_when_within_budget(self):
        agent = make_agent(
            "SYSTEM-指令-永不裁剪-",
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "我是助手"},
                {"role": "tool", "tool_call_id": "c1", "content": "read done"},
            ],
        )
        cm = ContextManager(agent, total_budget=2000)
        prompt, meta = cm.build("今天天气如何")

        assert meta["over_budget"] is False
        assert meta["degradation"] == []
        assert "SYSTEM" in prompt
        assert "今天天气如何" in prompt
        assert "你好" in prompt
        assert "read done" in prompt

    def test_full_render_with_large_budget(self):
        tools = [{"role": "tool", "tool_call_id": f"c{i}", "content": "R" * 100} for i in range(10)]
        chat = [{"role": "user", "content": f"问题{i}"} for i in range(5)]
        agent = make_agent("S" * 100, tools + chat)
        cm = ContextManager(agent, total_budget=100000)

        prompt, meta = cm.build("新请求")
        assert meta["over_budget"] is False
        assert "R" * 100 in prompt          # 全部工具结果都在
        assert "问题4" in prompt            # 全部对话都在
        assert meta["degradation"] == []


# ---------------------------------------------------------------------------
# 三级渐进式降级
# ---------------------------------------------------------------------------
class TestProgressiveDegradation:
    def _build_context(self):
        """system 100 + 工具 10×300 + 对话 15 条(>CHAT_KEEP 12) + 用户。"""
        tools = [
            {"role": "tool", "tool_call_id": f"c{i}", "content": "R" * 300}
            for i in range(10)
        ]
        chat = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "对话" * 30}
            for i in range(15)
        ]
        agent = make_agent("S" * 100, tools + chat)
        cm = ContextManager(agent, total_budget=100000)
        return cm, tools, chat

    # --- 单元级: 直接验证各级降级行为 ---
    def test_l1_tool_summarize(self):
        cm, tools, _ = self._build_context()
        full = cm._full_tools(tools)
        out = cm._l1_summarize_tools(tools)
        assert "已折叠 2 条较早工具结果" in out   # 10 条 - 保留 8 = 折叠 2
        assert len(out) < len(full)               # L1 后工具段更短

    def test_l2_fifo_drop(self):
        cm, _, chat = self._build_context()
        full = cm._full_chat(chat)
        out = cm._l2_fifo_chat(chat)
        assert "已丢弃 3 条最早对话" in out       # 15 条 - 保留 12 = 丢弃 3
        assert len(out) < len(full)               # L2 后对话段更短

    def test_l3_force_cut(self):
        cm = ContextManager(agent=type("A", (), {})(), total_budget=100000)
        sys_sec = Section("system", 100, pinned=True)
        sys_sec.rendered = "S" * 100
        tool_sec = Section("tools", 3000, pinned=False)
        tool_sec.rendered = "T" * 2000
        chat_sec = Section("chat", 3000, pinned=False)
        chat_sec.rendered = "C" * 2000

        cm._l3_force_cut([sys_sec, tool_sec, chat_sec], remain=500)
        assert len(tool_sec.rendered) + len(chat_sec.rendered) <= 500
        assert tool_sec.outcome == "L3"
        assert chat_sec.outcome == "L3"
        assert len(sys_sec.rendered) == 100       # 驻留区不动

    # --- 集成级: build 触发降级链路 ---
    def test_build_triggers_l1_and_l2(self):
        cm, _, _ = self._build_context()
        cm.total_budget = 2000   # system 100 + user 63, 剩余 1837 给非驻留 → 触发 L1+L2
        prompt, meta = cm.build("U" * 60)
        assert "L1_tool_summarize" in meta["degradation"]
        assert "L2_fifo_drop" in meta["degradation"]
        assert "已折叠" in prompt
        assert "已丢弃" in prompt

    def test_build_l3_fallback(self):
        cm, _, _ = self._build_context()
        user_msg = "U" * 60
        cm.total_budget = 500    # 极小预算 → L3 兜底
        prompt, meta = cm.build(user_msg)
        assert "L3_force_cut" in meta["degradation"]
        # 非驻留段被大幅压缩
        assert meta["sections"]["tools"]["chars"] < 3000
        assert meta["sections"]["chat"]["chars"] < 1300
        # 驻留区完整保留
        assert "S" * 100 in prompt
        assert f"User: {user_msg}" in prompt

    def test_full_degradation_chain(self):
        """极小预算下完整触发 L1 → L2 → L3 三阶链路。"""
        cm, _, _ = self._build_context()
        cm.total_budget = 500
        prompt, meta = cm.build("U" * 60)
        assert meta["degradation"] == ["L1_tool_summarize", "L2_fifo_drop", "L3_force_cut"]
        # 驻留区完整
        assert "S" * 100 in prompt
        assert "User: " in prompt


# ---------------------------------------------------------------------------
# 驻留区保护
# ---------------------------------------------------------------------------
class TestPinnedSections:
    def test_system_and_user_never_cut(self):
        tools = [{"role": "tool", "tool_call_id": "c1", "content": "T" * 300}]
        chat = [{"role": "user", "content": "C" * 300}]
        system_text = "SYSTEM核心指令" * 20
        agent = make_agent(system_text, tools + chat)
        cm = ContextManager(agent, total_budget=50)   # 极小预算

        prompt, meta = cm.build("用户关键请求" * 10)
        sections = meta["sections"]

        # 驻留区完整保留
        assert system_text in prompt
        assert "用户关键请求" in prompt
        assert sections["system"]["pinned"] is True
        assert sections["user"]["pinned"] is True

        # 非驻留区（tools/chat）被裁剪/降级
        assert sections["tools"]["outcome"] in ("L1", "L3")
        assert sections["chat"]["outcome"] in ("L2", "L3")

    def test_l3_force_cut_keeps_pinned(self):
        system = Section("system", 100, pinned=True)
        system.rendered = "系统指令内容" * 10          # 50 字符
        user = Section("user", 0, pinned=True)
        user.rendered = "U" * 50                       # 50 字符
        tool = Section("tools", 100, pinned=False)
        tool.rendered = "T" * 100                      # 100 字符

        cm = ContextManager(agent=type("A", (), {})(), total_budget=300)
        # 非驻留段可用预算 < 其原文长度(100), 确保 L3 实际发生截断
        remain = 60
        cm._l3_force_cut([system, tool, user], remain)

        assert "系统指令内容" in system.rendered      # 驻留: 完整
        assert len(user.rendered) == 50               # 驻留: 完整
        assert len(tool.rendered) < 100               # 非驻留: 被截断
        assert tool.outcome == "L3"


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_extreme_small_budget_no_crash(self):
        """极小预算下 L3 兜底不应崩溃, 驻留区仍完整。"""
        tools = [{"role": "tool", "tool_call_id": f"c{i}", "content": "T" * 200} for i in range(6)]
        chat = [{"role": "user", "content": "C" * 200} for _ in range(4)]
        agent = make_agent("S" * 100, tools + chat)
        cm = ContextManager(agent, total_budget=20)

        prompt, meta = cm.build("用户请求")
        assert "S" * 100 in prompt        # 系统驻留完整
        assert "用户请求" in prompt        # 用户驻留完整
        assert meta["over_budget"] is True  # 驻留区本身就超预算, 标记但保命

    def test_empty_history(self):
        """无历史时不触发任何降级, 只有 system + user。"""
        agent = make_agent("SYSTEM", [])
        cm = ContextManager(agent, total_budget=100)
        prompt, meta = cm.build("hello")

        assert meta["degradation"] == []
        assert "SYSTEM" in prompt
        assert "hello" in prompt
        assert "Tools: (none)" in prompt
        assert "Chat: (empty)" in prompt

    def test_role_enum_support(self):
        """Session 里 role 可能是 Role 枚举, 应能被正确识别。"""
        from core.session import Role

        agent = make_agent(
            "SYSTEM",
            [
                {"role": Role.USER, "content": "你好"},
                {"role": Role.ASSISTANT, "content": "我是助手"},
                {"role": Role.TOOL, "tool_call_id": "t1", "content": "结果"},
            ],
        )
        cm = ContextManager(agent, total_budget=1000)
        prompt, meta = cm.build("新的问题")

        assert meta["degradation"] == []
        assert "你好" in prompt and "新的问题" in prompt