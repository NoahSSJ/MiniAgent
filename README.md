# MiniAgent V2

从零开发的轻量级多模型路由 Agent。

> v1 完整保留在 master 分支。

## 项目简介

MiniAgent V2 是一个基于 **Ollama 本地模型 + DeepSeek 云端模型** 的轻量级 Agent 框架，核心设计：

- **模型路由**：用 `qwen2.5:0.5b` 做输入分类，按任务类型将请求路由到不同规格的模型（1.5b / 7b / DeepSeek API），在速度与质量之间取得平衡。
- **Agent 闭环**：主 Agent（7b）支持 Function Calling（工具调用），可读取/列出工作区文件。
- **委派机制（Delegate）**：主 Agent 遇到复杂调查任务时，可委派给**只读子 Agent** 去调查，结果以纯文本回流到父上下文（限制 4000 字符），子 Agent 步数受限、持久化隔离，避免递归失控。

## 环境要求

| 组件 | 要求 |
|------|------|
| Python | 3.14 |
| Ollama | 本地已安装并拉取 `qwen2.5:0.5b`、`qwen2.5:1.5b`、`qwen2.5:7b` |
| DeepSeek API | 可选（复杂推理兜底），需 `.env` 配置 Key |

## 环境安装

### 1. 创建 Conda 环境

```bash
conda create -n mini-agent-0 python=3.14 -y
conda activate mini-agent-0
```

### 2. 安装项目依赖

```bash
pip install -e .            # 仅运行依赖
pip install -e ".[dev]"     # 含测试依赖 (pytest)
```

### 3. 配置环境变量 `.env`

复制并填写（`main.py` 通过 `load_dotenv()` 自动加载）：

```bash
# 可选：DeepSeek 云端强模型（复杂推理使用）
DS_API_KEY=sk-your-key
DS_MODEL=deepseek-chat
DS_BASE_URL=https://api.deepseek.com
```

## 快速开始

### 运行 Agent 主程序

```bash
python main.py
```

主程序会初始化模型池：

| 模型 | 用途 |
|------|------|
| `qwen2.5:0.5b` | 输入分类器（仅做分类，最快） |
| `qwen2.5:1.5b` | 闲聊 / 简单事实 / 生活 |
| `qwen2.5:7b`  | 代码 / 文本创作 / Agent 主模型（支持工具） |
| `deepseek-chat` | 复杂深度推理（云端 API） |

### 路由分类说明

`core/router.py` 内置关键词规则（快速命中即返回，不调分类模型），规则未命中才调用 0.5b 分类器：

| 类别 | 场景 | 路由模型 |
|------|------|----------|
| `chat` | 闲聊/问候 | 1.5b |
| `simple` | 简单事实/常识 | 1.5b |
| `life` | 生活/日常建议 | 1.5b |
| `code` | 代码/编程/Bug | 7b |
| `creation` | 文本创作/写作 | 7b |
| `complex` | 复杂分析/深度推理 | DeepSeek API |
| `fallback` | 未识别/分类失败 | 7b（保守兜底） |

### 启动 Web 代理服务（可选）

`core/route.py` 提供 FastAPI 服务，代理 Ollama API：

```bash
python core/route.py    # 默认 http://127.0.0.1:8000
```

- `GET /models` — 列出 Ollama 已安装模型
- `POST /gen` — 调用 Ollama `/api/generate`

## Agent 委派机制

```mermaid
flowchart LR
    A[主Agent 7b] -->|delegate_task| B[只读子Agent]
    B -->|read_file_tool| C[工作区文件]
    C -->|纯文本结论| B
    B -->|≤4000字符| A
```

- 主 Agent 固定使用 7b（小模型不支持 Function Calling）。
- 子 Agent：`read_only=True`、`max_steps=3`，不注册 `delegate_task`（防递归），Session 不持久化。
- 委派结果严格裁剪至 4000 字符，防止撑爆父 Agent 上下文。

## 项目结构

```
MiniAgent/
├── main.py                 # 主程序入口: 模型池 + 路由表 + Agent
├── pyproject.toml          # 项目元数据与依赖
├── core/
│   ├── agent.py            # MiniAgent: Session/工具/委派编排
│   ├── llm.py              # MiniLLM: OpenAI 兼容客户端封装
│   ├── router.py           # ModelRouter: 关键词+模型分类路由
│   ├── route.py            # FastAPI 代理服务 (Ollama)
│   ├── tools.py            # 工具注册/校验/执行 + 内置工具
│   ├── session.py          # Session/消息/Role 定义与持久化
│   ├── prompt.py           # 系统提示词构建
│   ├── context_v1.py       # 上下文管理 v1
│   ├── context_v2.py       # 上下文管理 v2
│   ├── workspace.py        # 工作区上下文（路径安全）
│   ├── logger.py           # 全局日志 (loguru, double)
│   └── example.py          # 示例
├── tests/
│   ├── test_context.py     # 上下文测试
│   └── test_delegate.py    # 委派机制测试
└── web/                    # Web 资源（预留）
```

## 运行测试

```bash
pytest
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DS_API_KEY` | 空 | DeepSeek API Key |
| `DS_MODEL` | `deepseek-chat` | 云端强模型名 |
| `DS_BASE_URL` | `https://api.deepseek.com` | 云端 API 地址 |
| `APP_ENV` | `development` | `production` 时控制台静默 |
| `LOG_LEVEL` | 开发 `DEBUG` / 生产 `INFO` | 日志级别 |
| `LOG_DIR` | `logs/` | 日志目录 |

## 常见问题

**Q: 请求 localhost:11434 报 502？**
A: `MiniLLM` 使用 `httpx.Client(trust_env=False)` 禁用系统代理，确保本地 Ollama 请求不被 Clash 等代理劫持。

**Q: 使用 conda 环境后提示缺少依赖？**
A: 执行 `pip install -e ".[dev]"` 一次性安装全部运行依赖与测试依赖。