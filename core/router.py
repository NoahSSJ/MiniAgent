"""
模型路由: 用 qwen2.5:0.5b 对用户输入做快速分类, 再根据类别路由到不同模型。

分类 → 模型映射:
  - chat     闲聊/问候          → 1.5b   (qwen2.5:1.5b)
  - simple   简单事实/常识问答   → 1.5b   (qwen2.5:1.5b)
  - life     生活/日常建议      → 1.5b   (qwen2.5:1.5b)
  - code     代码/编程/Bug修复  → 7b     (qwen2.5:7b)
  - creation 文本创作/写作      → 7b     (qwen2.5:7b)
  - complex  复杂分析/长文档/需要深度推理
                             → 7b     (qwen2.5:7b), 可通过 max_tokens/温度兜底
  - fallback 未识别 / 分类失败  → 7b     (保守选强模型)
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .llm import MiniLLM
from .logger import logger

# 分类标签
CATEGORY_CHAT = "chat"
CATEGORY_SIMPLE = "simple"
CATEGORY_LIFE = "life"
CATEGORY_CODE = "code"
CATEGORY_CREATION = "creation"
CATEGORY_COMPLEX = "complex"
CATEGORY_FALLBACK = "fallback"

# 关键词规则: (正则, 类别) — 命中即返回, 不调用分类模型
RULE_PATTERNS: list[tuple[str, str]] = [
    # 复杂分析类: 优先匹配(含"分析/原理/推导/数学"等强信号)
    (r"(详细|深入|深度|全面|完整)?分析|原理|数学推导|论证|证明|"
     r"量子|哲学|宏观经济|论文|报告|综述|读后感|解读", CATEGORY_COMPLEX),
    # 代码类
    (r"写.{0,6}(代码|程序|脚本|函数|接口)|编程|bug|报错|异常|重构|debug"
     r"|python|java|javascript|typescript|golang|rust|sql|docker|git "
     r"|部署|命令行|脚本|正则|算法|leetcode|api\b|html|css|c\+\+|\bjs\b|\bgo\b", CATEGORY_CODE),
    # 创作类
    (r"写.{0,10}(文章|故事|小说|诗|诗篇|文案|作文|剧本|歌词)|润色|翻译|"
     r"缩写|扩写|起个题|拟定提纲|文案策划", CATEGORY_CREATION),
    # 生活类
    (r"吃|菜谱|做饭|食谱|感冒|养生|运动|健身|减肥|护肤|穿搭|旅游|酒店|"
     r"机票|购物|买|装修|家居|育儿|婚姻|人际关系|生活", CATEGORY_LIFE),
    # 闲聊类
    (r"^你好|^嗨|^哈喽|^hello|^hi\b|早上好|晚上好|中午好|谢谢|感谢|"
     r"再见|拜拜|辛苦了|心情|怎么样$|你是谁|你会什么", CATEGORY_CHAT),
    # 简单事实类
    (r"谁是|什么是|是什么|为什么|多少岁|成立于|首都是|面积|人口|"
     r"是谁|哪个国家|哪一年|历史|定义|含义", CATEGORY_SIMPLE),
]

VALID_CATEGORIES = {
    CATEGORY_CHAT, CATEGORY_SIMPLE, CATEGORY_LIFE,
    CATEGORY_CODE, CATEGORY_CREATION, CATEGORY_COMPLEX,
    CATEGORY_FALLBACK,
}

CLASSIFY_SYSTEM_PROMPT = """\
你是输入分类器。根据用户的输入,只输出一个 JSON 对象(不要输出任何其他内容),格式:
{"category": "<分类>"}

分类定义:
- "chat": 问候、闲聊、日常寒暄、情感倾诉
- "simple": 简单事实问答、常识、百科类问题、一句话能答完的问题
- "life": 生活建议、健康、美食、旅游、家居、购物等日常实用问题
- "code": 编程相关: 写代码、改代码、修 bug、解释代码、部署、命令行、报错排查
- "creation": 文本创作: 写文章、写故事、写诗、翻译、润色、提纲、文案
- "complex": 深度分析、复杂推理、数学推导、多步骤论证的疑难问题;涉及"分析/原理/推理/论证/论文/报告/解读"等词也归此类
- "fallback": 无法归入以上任何类别

只输出 JSON,例如 {"category": "chat"}。"""


class ModelRouter:
    """模型路由: 分类 + 分发"""

    def __init__(
        self,
        classifier: MiniLLM,
        route_table: dict[str, MiniLLM],
        *,
        default_model: Optional[MiniLLM] = None,
    ) -> None:
        self.classifier = classifier
        self.route_table = route_table
        self.default_model = default_model or self._pick_default(route_table)

    @staticmethod
    def _pick_default(route_table: dict[str, MiniLLM]) -> MiniLLM:
        # 找不到 default_model 时, 优先用 7b(强模型)兜底
        for key in (CATEGORY_CODE, CATEGORY_COMPLEX, CATEGORY_CREATION, CATEGORY_FALLBACK):
            if key in route_table:
                return route_table[key]
        # 空表兜底: 返回 None(由调用方处理)或抛明确错误
        if not route_table:
            return None
        return next(iter(route_table.values()))

    # ------------------------------------------------------------
    # 分类
    # ------------------------------------------------------------
    def classify(self, user_input: str) -> str:
        """先跑关键词规则(快/稳), 规则未命中才用分类器模型"""
        if not user_input or not user_input.strip():
            return CATEGORY_FALLBACK

        rule_category = self._classify_by_rule(user_input)
        if rule_category is not None:
            return rule_category

        messages = [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
        try:
            resp = self.classifier.chat(
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = (resp.content or "").strip()
            category = self._parse_category(raw)
            return category if category in VALID_CATEGORIES else CATEGORY_FALLBACK
        except Exception as e:
            logger.warning(f"[router] 分类失败, 回退到 default: {e}")
            return CATEGORY_FALLBACK

    @staticmethod
    def _classify_by_rule(text: str) -> Optional[str]:
        """关键词规则匹配, 未命中返回 None"""
        for pattern, category in RULE_PATTERNS:
            if re.search(pattern, text, re.I):
                return category
        return None

    @staticmethod
    def _parse_category(raw: str) -> str:
        """从模型输出里提取 category, 容忍 ```json 代码块 / 多余文字"""
        # 先尝试整体 JSON 解析
        text = raw.strip()
        # 去掉 markdown 代码块
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if m:
            text = m.group(1).strip()
        # 提取 JSON 对象
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                cat = data.get("category")
                if isinstance(cat, str):
                    return cat.strip().lower()
            except json.JSONDecodeError:
                pass
        # 最后尝试正则直接抓 category 后冒号引号里的词
        m = re.search(r'category["\']?\s*[:：]\s*["\']?([a-zA-Z_]+)', text)
        if m:
            return m.group(1).strip().lower()
        return CATEGORY_FALLBACK

    # ------------------------------------------------------------
    # 路由 + 推理
    # ------------------------------------------------------------
    def chat(
        self,
        user_input: str,
        messages: list[dict] | None = None,
        tools_schema: list = None,
        classify_flag: bool = True,
        **kwargs,
    ):
        """
        根据用户输入分类 → 选模型 → 调用模型。

        :param user_input:  用于分类的原始输入(首次用户消息)
        :param messages:    完整消息历史(传给模型)
        :param classify_flag: 是否执行分类路由; False 则走 default_model
        """
        category = (
            self.classify(user_input)
            if classify_flag and user_input
            else CATEGORY_FALLBACK
        )
        model = self.route_table.get(category, self.default_model)

        history = messages if messages is not None else [{"role": "user", "content": user_input}]
        return model.chat(
            messages=history,
            tools_schema=tools_schema,
            **kwargs,
        )
