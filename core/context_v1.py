"""Mini 版上下文预算管理器。

为什么存在这个模块？
大模型上下文窗口是有限的（例如 8K/32K/128K token），而一个 agent 会话会不断累积：
系统指令、工具执行结果、对话历史……如果一股脑全塞进模型，很快就会触达窗口上限，
导致请求报错或模型"记不住"最近的关键信息。

本模块解决的核心问题：**如何在有限预算内，尽可能保住"最有价值"的上下文**。

核心思路（与 example.py / CoreCoder 的思想一致，但刻意保持精简）：
1. **分区分段**：把上下文划分为独立 Section —— 系统指令 / 工具历史 / 对话历史 / 当前用户输入。
   每个 Section 拥有独立预算，互不干扰。
2. **渐进式降级**：当整体超预算时，按"信息价值从低到高"的顺序逐层压缩：
   L1 工具结果摘要压缩 → L2 丢弃最早历史(FIFO) → L3 强制截断兜底。
   每一步都以"尽量少损失信息"为原则，能不动高级信息就不动。
3. **永久驻留区**：系统指令与当前用户输入标记为 pinned（驻留），
   它们绝不参与任何一层裁剪，保障核心指令与最新请求零丢失。

模块对外只暴露一个核心入口：``ContextManager.build(user_message)``，
返回 (prompt 文本, metadata 元数据)，metadata 记录了每段原始长度、降级明细，
便于日后追踪/审计这一帧 prompt 是怎么被拼出来的。
"""

from __future__ import annotations  # 让类型注解支持延迟求值（模块级可用 | 语法）


# ---------------------------------------------------------------------------
# 全局默认配置（可通过 ContextManager 构造参数覆盖）
# ---------------------------------------------------------------------------

# 默认整体预算上限（字符数）。注意：这是"字符数"而非 token 数，
# 是 example.py 风格的简化近似；中英混合内容大致可按 3~4 字符 ≈ 1 token 换算。
DEFAULT_TOTAL_BUDGET = 8000

# 各非驻留段的默认预算分配。
#   - system: 系统指令预算（但它同时是 pinned=驻留，这里标记的是其默认最大体量预期）
#   - tools : 工具历史段预算，工具结果通常又长又碎，给足空间但压缩时优先牺牲它
#   - chat  : 对话历史段预算，普通问答往来
# "user" 段不设预算（0），因为它永远驻留、永不裁剪，不需要分配预算。
DEFAULT_BUDGETS = {
    "system": 2000,   # 系统指令（驻留）
    "tools": 3000,    # 工具历史
    "chat": 3000,     # 对话历史
}

# --- L1 工具结果摘要压缩的参数 ---
TOOL_SUMMARY_HEAD = 120   # 每条保留下来的工具结果，最多保留前 N 字符
TOOL_SUMMARY_KEEP = 8     # 最多保留最近 N 条工具结果；更早的工具结果折叠成一行提示

# --- L2 对话历史 FIFO 丢弃的参数 ---
CHAT_KEEP = 12            # 对话历史最多保留最近 N 条（更早的按 FIFO 丢弃）
CHAT_LINE_LIMIT = 200     # 保留下来的对话，单条最大长度（超出则截断）


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _tail(text, limit):
    """把文本截断到 limit 长度，并在末尾追加省略号 "..."。

    - limit <= 0         -> 返回空串（无预算可写）
    - 文本本来就 <= limit -> 原样返回（不需要截断）
    - limit > 3          -> 截断到 limit-3 再补 "..."，保证总长 == limit
    - limit <= 3         -> 空间太小，直接硬切 limit 个字符，不加省略号
    """
    text = str(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..." if limit > 3 else text[:limit]


# ---------------------------------------------------------------------------
# Section: 一个上下文分区
# ---------------------------------------------------------------------------

class Section:
    """上下文分区。

    每个 Section 是一段独立的上下文：系统指令、工具历史、对话历史、用户请求……
    - budget : 该段的预算上限（字符数）；user 段为 0 表示"不设预算，永不裁剪"
    - pinned : 是否永久驻留。pinned=True 的段在 L1/L2/L3 降级时完全不参与，
               保证系统指令与当前用户请求永远完整送达模型。
    - rendered: 该段当前渲染出来的最终文本（经各级降级后的结果）
    - outcome : 记录这一段最终经历了哪一级降级，便于审计：
                "none"（未降级）/ "L1" / "L2" / "L3"
    """

    def __init__(self, name, budget, pinned=False):
        self.name = name                        # 段名: system / tools / chat / user
        self.budget = budget                    # 预算上限（字符数）
        self.pinned = pinned                    # 是否驻留（永不裁剪）
        self.rendered = ""                      # 渲染结果文本
        self.outcome = "none"                   # 降级记录: none / L1 / L2 / L3

    @property
    def chars(self):
        """当前渲染文本的字符数（便捷属性）。"""
        return len(self.rendered)


# ---------------------------------------------------------------------------
# ContextManager: 上下文预算管理器（核心类）
# ---------------------------------------------------------------------------

class ContextManager:
    def __init__(self, agent, total_budget=DEFAULT_TOTAL_BUDGET, budgets=None):
        """初始化。

        - agent        : 持有 prompt / session 的 agent 对象。
                         本模块通过 duck-typing 兼容 MiniAgent、Pico、或任意
                         提供 .prompt / .session / .prefix 的对象。
        - total_budget : 整体预算（字符数），超限才会触发降级。
        - budgets      : 可选，覆盖各段的默认预算，如 {"tools": 5000}。
        """
        self.agent = agent
        self.total_budget = int(total_budget)          # 强制转 int，防止外部传入 "8000" 等字符串
        self.budgets = dict(DEFAULT_BUDGETS)           # 深拷贝默认预算，避免污染全局默认值
        if budgets:
            # 仅覆盖用户显式传的段，其余保持默认
            self.budgets.update({str(k): int(v) for k, v in budgets.items()})

    # ------------------------------------------------------------------
    # 取数：从 agent 身上安全地拿系统指令 / 历史
    # ------------------------------------------------------------------

    def _system_text(self):
        """获取系统指令文本。

        优先从 agent.prompt.get_system_prompt().prompt_text 取
        （MiniAgent 的 PromptManager 形态），失败则回退到 agent.prefix
        （Pico 形态）。全程 try/except，保证即使 prompt 结构异常也不崩溃。
        """
        prompt = getattr(self.agent, "prompt", None)
        if prompt is not None and hasattr(prompt, "get_system_prompt"):
            try:
                return str(prompt.get_system_prompt().prompt_text)
            except Exception:
                pass            # 取不到就继续往下走回退逻辑
        return str(getattr(self.agent, "prefix", ""))

    def _history_items(self):
        """获取会话历史消息列表。

        兼容两种形态：
        - MiniAgent : session 是 Session 对象，历史在 .history 属性
        - Pico      : session 本身就是 dict，历史在 {"history": [...]} 里
        """
        session = getattr(self.agent, "session", None)
        if session is None:
            return []
        if isinstance(session, dict):
            return list(session.get("history", []) or [])
        return list(getattr(session, "history", []) or [])

    @staticmethod
    def _role(item):
        """把消息的 role 字段规整成字符串。

        MiniAgent 的 Session 里 role 可能是 Role 枚举（如 Role.USER），
        而枚举实例没法直接和字符串比较，这里统一取出 .value。
        """
        role = item.get("role")
        return role.value if hasattr(role, "value") else role

    @staticmethod
    def _content(item):
        """安全取消息文本，None 归一为空串。"""
        return str(item.get("content") or "")

    def _split(self, items):
        """把历史拆成工具段(tools)与对话段(chat)两个列表。

        工具消息（role=="tool"）单独成段，方便 L1 单独针对工具结果做摘要压缩；
        其余（user/assistant）归为对话段，供 L2 做 FIFO 丢弃。
        """
        tools, chat = [], []
        for item in items:
            (tools if self._role(item) == "tool" else chat).append(item)
        return tools, chat

    # ------------------------------------------------------------------
    # 渲染：把原始消息转成可读文本（未压缩 / 压缩两种）
    # ------------------------------------------------------------------

    def _full_tools(self, tools):
        """全量渲染工具段：所有工具结果原样列出（未压缩）。

        格式：每行 "[tool:工具调用id] 结果内容"。
        工具调用 id 保留是为了让模型能对应上"哪个工具调用的结果"。
        """
        if not tools:
            return "Tools: (none)"
        body = "\n".join(
            f"[tool:{item.get('tool_call_id', '?')}] {self._content(item)}"
            for item in tools
        )
        return f"Tools:\n{body}"

    def _full_chat(self, chat):
        """全量渲染对话段：所有 user/assistant 消息原样列出（未压缩）。"""
        if not chat:
            return "Chat: (empty)"
        body = "\n".join(
            f"[{self._role(item)}] {self._content(item)}" for item in chat
        )
        return f"Chat:\n{body}"

    # ------------------------------------------------------------------
    # 三阶渐进式降级（信息价值从低到高，越后面损失越大、越少触发）
    # ------------------------------------------------------------------

    def _l1_summarize_tools(self, tools):
        """L1: 工具结果摘要压缩。

        为什么先压缩工具段？
        工具结果（如 read_file 的全文、run_shell 的 stdout）往往又长又琐碎，
        但真正对后续决策有用的只是其中一小部分。这是信息密度最低、最值得先压缩的段。

        做法：
        - 只保留最近 TOOL_SUMMARY_KEEP(8) 条工具结果；
        - 每条再截断到 TOOL_SUMMARY_HEAD(120) 字符；
        - 更早的工具结果折叠为一行："[tool] 已折叠 N 条较早工具结果"，
          用一行提示代替几百上千字符，让模型知道"这些工具被调用过、但细节被省了"。
        """
        # older: 超出保留额度、需要折叠成一行提示的旧工具结果
        older = tools[:-TOOL_SUMMARY_KEEP] if len(tools) > TOOL_SUMMARY_KEEP else []
        # recent: 最近 TOOL_SUMMARY_KEEP 条，保留（但仍截断到 120 字符）
        recent = tools[-TOOL_SUMMARY_KEEP:]
        lines = []
        if older:
            lines.append(f"[tool] 已折叠 {len(older)} 条较早工具结果")
        for item in recent:
            lines.append(
                f"[tool:{item.get('tool_call_id', '?')}] "
                f"{_tail(self._content(item), TOOL_SUMMARY_HEAD)}"
            )
        return f"Tools:\n" + "\n".join(lines) if lines else "Tools: (none)"

    def _l2_fifo_chat(self, chat):
        """L2: 丢弃最早历史（FIFO 滑动窗口）。

        为什么 L2 才动对话段？
        对话历史的信息价值比工具结果高（承载了用户意图与助手推理），
        所以只在 L1 仍不够时才压缩它。

        做法：
        - 只保留最近 CHAT_KEEP(12) 条对话（FIFO：最早的自然被挤出窗口）；
        - 每条单行截断到 CHAT_LINE_LIMIT(200) 字符；
        - 若真的丢弃了旧对话，加一行提示 "[chat] 已丢弃 N 条最早对话"，
          让模型意识到历史被截断过，避免它以为漏了什么。
        """
        recent = chat[-CHAT_KEEP:]            # 滑动窗口，取最后 CHAT_KEEP 条
        lines = []
        if len(chat) > CHAT_KEEP:
            lines.append(f"[chat] 已丢弃 {len(chat) - CHAT_KEEP} 条最早对话")
        for item in recent:
            lines.append(
                f"[{self._role(item)}] {_tail(self._content(item), CHAT_LINE_LIMIT)}"
            )
        return f"Chat:\n" + "\n".join(lines) if lines else "Chat: (empty)"

    def _l3_force_cut(self, sections, remain):
        """L3: 强制截断兜底（最后手段）。

        当 L1 + L2 都做完了仍超预算时，只能硬性砍非驻留段的长度。
        驻留区（system + user）完全排除在分配之外——它们永不被裁。

        做法：
        - 只取"非驻留且有内容"的段参与分配（pinned 的 system/user 被跳过）；
        - 按各段当前长度的比例分配剩余预算 remain（长得越多的砍得越多，更公平）；
        - 关键细节：最后一段的预算 = remain - 已分配的部分（而不是直接拿 remain），
          否则前面段分配走一部分后，最后一段仍拿满 remain 会导致总长超预算
          （这是本函数曾修过的真实 bug）；
        - budget 下限 clamp 到 0，防止剩余预算为负时抛错。
        所有被 L3 处理的段 outcome 标记为 "L3"。
        """
        pool = [sec for sec in sections if not sec.pinned and sec.chars > 0]
        total = sum(sec.chars for sec in pool) or 1   # or 1 防止除零
        allocated = 0                                  # 已分配的预算累计
        for i, sec in enumerate(pool):
            if i == len(pool) - 1:
                budget = remain - allocated   # 最后一段吃掉真正剩余的预算
            else:
                budget = int(remain * sec.chars / total)  # 按长度比例分配
            budget = max(0, budget)
            sec.rendered = _tail(sec.rendered, budget)
            sec.outcome = "L3"
            allocated += len(sec.rendered)     # 累计真实占用的长度

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def build(self, user_message):
        """组装一帧上下文。这是模块的唯一对外入口。

        流程：
        1. 从 agent 取系统指令、拆历史为工具/对话两段，构建 4 个 Section；
        2. 先按"全量渲染"估算总长，若超总预算：
           L1 摘要工具段 →（仍超）L2 FIFO 丢对话 →（仍超）L3 硬截断非驻留段；
        3. 组装所有 Section 为最终 prompt；
        4. 返回 (prompt, metadata)，metadata 记录超限/降级明细，便于审计。

        返回值：
        - prompt  : 最终发给模型的文本（纯文本形态，可与 example.py 输出对齐）
        - metadata: dict，包含总长、是否超预算、降级顺序、每段的预算/长度/驻留/降级级别
        """
        user_message = str(user_message)

        # --- 构建 4 个 Section ---
        # system: 驻留，内容来自系统提示词
        system = Section("system", self.budgets["system"], pinned=True)
        system.rendered = self._system_text()

        # user: 驻留，内容 = 当前用户请求。预算 0 表示不设上限、永不裁剪
        user = Section("user", 0, pinned=True)
        user.rendered = f"User: {user_message}"

        # tools / chat: 非驻留，先全量渲染
        tools, chat = self._split(self._history_items())
        tools_sec = Section("tools", self.budgets["tools"])
        chat_sec = Section("chat", self.budgets["chat"])
        tools_sec.rendered = self._full_tools(tools)
        chat_sec.rendered = self._full_chat(chat)

        sections = [system, tools_sec, chat_sec, user]
        degradation = []   # 记录触发了哪几级降级（按触发顺序）

        # L1: 工具结果摘要压缩。仅当"超预算且有工具结果可压缩"时才执行
        if len(self._assemble(sections)) > self.total_budget and tools:
            tools_sec.rendered = self._l1_summarize_tools(tools)
            tools_sec.outcome = "L1"
            degradation.append("L1_tool_summarize")

        # L2: 丢弃最早历史（FIFO）。仅当 L1 后仍超预算且有对话可丢弃时执行
        if len(self._assemble(sections)) > self.total_budget and chat:
            chat_sec.rendered = self._l2_fifo_chat(chat)
            chat_sec.outcome = "L2"
            degradation.append("L2_fifo_drop")

        # L3: 强制截断兜底。仅当 L1+L2 后仍超预算时执行；驻留区不受影响。
        #     剩余可用预算 = 总预算 - 驻留区(system+user)长度，全部分给非驻留段
        if len(self._assemble(sections)) > self.total_budget:
            remain = max(0, self.total_budget - len(system.rendered) - len(user.rendered))
            self._l3_force_cut(sections, remain)
            degradation.append("L3_force_cut")

        # --- 组装最终 prompt 并生成 metadata ---
        prompt = self._assemble(sections)
        metadata = {
            "prompt_chars": len(prompt),                     # 最终 prompt 字符数
            "total_budget": self.total_budget,               # 总预算
            "over_budget": len(prompt) > self.total_budget,  # 是否仍超预算
                                                             # （驻留区本身超时也会 True，但已尽力保核心）
            "degradation": degradation,                      # 触发的降级链路，如 ["L1", "L2", "L3"]
            "sections": {
                sec.name: {
                    "budget": sec.budget,    # 该段预算
                    "chars": sec.chars,      # 最终渲染长度
                    "pinned": sec.pinned,    # 是否驻留
                    "outcome": sec.outcome,  # 该段经历的降级级别
                }
                for sec in sections
            },
        }
        return prompt, metadata

    def _assemble(self, sections):
        """把各 Section 渲染文本按固定顺序拼接成最终 prompt。

        顺序刻意设计：稳定基线（system）在前，最新请求（user）在最后，
        历史在中间——与 example.py 的 SECTION_ORDER 语义一致。
        空段被过滤（避免出现连续的空白分隔）。
        """
        return "\n\n".join(sec.rendered for sec in sections if sec.rendered).strip()