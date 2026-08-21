# 项目架构文档

## 整体概览

中医问诊智能体是一个基于 Flask + LangGraph 的对话式医疗 AI 应用，采用模块化分层架构。各子模块按职责独立，通过配置中心与可观测性契约协作，不直接互相 import。

```
┌─────────────────────────────────────────────────────────┐
│                     应用入口层                            │
│   app.py (Flask Web)    main.py (CLI)    start.py       │
├─────────────────────────────────────────────────────────┤
│                   配置中心 (config_manager.py)            │
│   config.env → os.environ → 所有子模块共享               │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│  Agent   │   LLM    │  Memory  │  Tools   │ Knowledge   │
│  agent/  │  llm/    │ memory/  │ tools/   │ knowledge/  │
├──────────┴──────────┴──────────┴──────────┴─────────────┤
│              可观测性 (observability/)                     │
│   logging + JSONL 事件流 + 指标聚合                       │
├─────────────────────────────────────────────────────────┤
│              MCP 桥接 (mcp_bridge/)                       │
│   外部工具协议适配（高德地图等）                            │
├─────────────────────────────────────────────────────────┤
│              语音模块 (voice.py)                           │
│   ASR (Paraformer) + TTS (CosyVoice)                     │
└─────────────────────────────────────────────────────────┘
```

## 目录结构

```
Medical-Agent/
├── app.py                  # Flask Web 入口，路由定义与会话管理
├── main.py                 # CLI 交互入口
├── start.py                # 启动器（设置环境变量 + 拉起 app.py + 打开浏览器）
├── config_manager.py       # 配置中心：读取/写入/校验/注入环境变量
├── config.env              # 运行时配置（不入库，含 API Key）
├── config.env.example      # 配置模板
├── voice.py                # 语音模块：ASR + TTS 客户端
├── requirements.txt        # Python 依赖（Flask + requests + langgraph）
│
├── agent/                  # Agent 运行时
│   ├── factory.py          # 工厂：按环境变量切换 langgraph/classic
│   ├── graph.py            # LangGraph StateGraph 状态图编排
│   ├── controller.py       # 经典手写循环（回退路径）
│   ├── planner.py          # 纯规则决策器
│   ├── router.py           # 动作 → 自然语言回复映射
│   └── runtime_utils.py    # 跨运行时通用能力
│
├── llm/                    # LLM 调用层
│   ├── llm.py              # DeepSeek API 封装（含重试/降级）
│   └── prompt.py           # 系统提示词模板
│
├── memory/                 # 记忆系统
│   ├── memory.py           # ConversationMemory 编排层
│   ├── file_store.py       # JSON 文件持久化（原子写入）
│   ├── profile_store.py    # 长期用户画像
│   └── session_store.py    # 当前会话状态
│
├── tools/                  # 医学规则工具集
│   ├── protocol.py         # ToolResult 统一返回结构 + managed_tool 装饰器
│   ├── registry.py         # 工具发现与注册中心
│   ├── symptom_tool.py     # 症状抽取（含否定句/城市抽取）
│   ├── risk_tool.py        # 风险分层（读取外置 JSON 规则）
│   ├── guideline_tool.py   # 分诊指南生成
│   └── knowledge_tool.py   # 中医知识检索（接入 BM25）
│
├── knowledge/              # 中医知识层
│   ├── tcm_knowledge.py    # 知识加载 + 校验 + 对外接口
│   ├── retriever.py        # BM25/TF-IDF 检索器（零依赖）
│   └── data/               # 外置 JSON 知识数据
│       ├── syndrome_rules.json       # 证型辨证规则
│       ├── term_normalization.json   # 术语归一化
│       ├── red_flags.json            # 红旗信号清单
│       ├── risk_rules.json           # 风险分层规则
│       ├── chief_complaint_followup.json  # 主诉追问模板
│       ├── formulas.json             # 方剂数据
│       ├── health_advice.json        # 调护建议
│       └── faq.json                  # 常见问答
│
├── observability/          # 可观测性
│   ├── __init__.py         # 统一 re-export
│   ├── events.py           # JSONL 结构化事件流
│   ├── logger.py           # 统一日志配置
│   └── metrics.py          # 指标聚合
│
├── mcp_bridge/             # MCP 桥接层
│   ├── __init__.py
│   ├── client.py           # MCP 客户端（含重连/指标）
│   ├── adapter.py          # 工具适配层（input_schema 校验）
│   └── config.py           # MCP 配置管理
│
├── mcp_servers/            # MCP 服务端实现
│   └── hospital_locator.py # 附近医院检索（高德 POI）
│
├── evaluation/             # 评测框架
│   ├── run_eval.py         # 评测回放引擎
│   ├── compare_retrievers.py  # BM25 vs TF-IDF 对比
│   ├── smoke_real_llm.py   # 真实 LLM 冒烟测试
│   └── cases/              # 评测用例（15 个 JSON）
│
├── templates/
│   └── index.html          # 前端单页面（对话/配置/状态 Tab）
│
├── docs/                   # 文档
│   ├── PROJECT_ARCHITECTURE.md  # 本文件
│   └── images/             # 截图
│
└── logs/                   # 运行日志（不入库）
```

## 核心模块详解

### 1. 配置中心 (`config_manager.py`)

单一配置源，所有业务参数集中声明。

- **配置来源优先级**：`DEFAULT_CONFIG` (硬编码) → `config.env` (覆盖) → `os.environ` (运行时)
- **运行时注入**：`apply_config_to_environment()` 将配置写入 `os.environ`，下游模块只读环境变量
- **校验**：`save_config()` 写入前校验（provider 枚举、数值范围），非法配置无法落盘
- **密钥保护**：`DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` 留空时保留已保存值，不会被误清空
- **热重载**：`/api/config/reload` 触发重建 LLM、Agent 并清空会话缓存

### 2. Agent 运行时 (`agent/`)

两种运行时通过 `AGENT_RUNTIME` 环境变量切换，对外接口一致。

#### LangGraph 模式（默认）

```
extract → risk_assess → plan → {动作节点} → review → respond
```

- `graph.py`：`StateGraph` 定义有向图，`InMemorySaver` checkpointer 支持多轮隔离
- 动作节点：`risk_escalation` / `clarify_conflict` / `ask_followup_single` / `ask_followup_bundle` / `summarize_progress` / `request_pulse_input` / `final_advice`

#### Classic 模式（回退）

- `controller.py`：手写 for 循环，复用同一 Planner/Router/runtime_utils

#### 共享组件

- `planner.py`：纯规则决策器，依据缺失槽位/置信度/风险等级输出下一步动作
- `router.py`：`next_action` → `ActionResult`（含自然语言回复）
- `runtime_utils.py`：槽位抽取、响应渲染、序列化、memory 同步等通用能力

### 3. LLM 调用层 (`llm/`)

- `llm.py`：DeepSeek API 封装，支持 mock/deepseek/auto 三种 provider
- 自动重试一次（指数退避），捕获网络异常
- 未配置 API Key 时自动降级到 mock 模式
- `prompt.py`：系统提示词模板，含 RAG 知识注入段落

### 4. 记忆系统 (`memory/`)

以 `ConversationMemory` 为编排层，按生命周期拆分为两层存储：

| 层级 | 存储 | 内容 |
|---|---|---|
| 长期画像 | `profile_store.py` → JSON 文件 | 用户基本信息、体质特征、历史症状 |
| 会话状态 | `session_store.py` → JSON 文件 | 对话历史、case_state、矛盾检测标记 |

- 持久化底层 `file_store.py` 采用原子写入（先写 `.tmp` 再 `os.replace`）
- 所有对外读取返回 `deepcopy`，保证内存单一数据源

### 5. 工具集 (`tools/`)

统一通过 `managed_tool` 装饰器暴露 `ToolResult`，支持发现、注册、计时。

| 工具 | 职责 |
|---|---|
| `symptom_tool.py` | 症状抽取 + 否定句豁免 + 城市槽位抽取 |
| `risk_tool.py` | 风险分层（读取外置 `risk_rules.json`） |
| `guideline_tool.py` | 分诊指南生成 |
| `knowledge_tool.py` | 中医知识检索（接入 BM25 检索器） |

### 6. 知识层 (`knowledge/`)

- `tcm_knowledge.py`：启动时加载 `data/*.json`，schema 校验失败抛 `KnowledgeLoadError`
- `retriever.py`：零依赖 BM25/TF-IDF 检索器，中文二元分词 + ASCII 整词
- 知识数据全部外置为 JSON，修改无需改代码

### 7. 可观测性 (`observability/`)

- `events.py`：JSONL 结构化事件流（`llm_call`、`tool_call`、`voice_call` 等）
- `logger.py`：统一日志配置（`RotatingFileHandler`）
- `metrics.py`：从事件流聚合调试指标
- 前端 `/api/debug/trace` 实时展示执行轨迹

### 8. MCP 桥接 (`mcp_bridge/`)

- `client.py`：MCP 客户端，支持断线重连 + 指标暴露
- `adapter.py`：工具适配层，调用前按 `input_schema` 本地校验参数
- `config.py`：MCP 服务器注册配置
- 当前试点：`hospital_locator.py`（高德地图 POI 附近医院检索）

### 9. 语音模块 (`voice.py`)

- **ASR**：Paraformer 语音识别（DashScope API）
- **TTS**：CosyVoice 语音合成，长文本自动分段（单段 ≤400 字符）
- `test_connection()`：4 字短文本真实合成探针
- 事件埋点：`voice_call`（kind=asr/tts/tts_test）

### 10. 评测框架 (`evaluation/`)

- `run_eval.py`：回放 `cases/*.json` 多轮对话，逐项断言风险/槽位/回复
- `compare_retrievers.py`：BM25 vs TF-IDF 检索质量对比
- `smoke_real_llm.py`：真实 DeepSeek API 端到端冒烟
- 15 个评测用例覆盖：高风险、红旗、矛盾、否定句、脉诊等场景

## 数据流

一次完整的问诊对话流程：

```
用户输入 → Flask /chat
  → SessionCache 获取/创建 Agent 实例
    → agent.run(user_message)
      → [extract] symptom_tool 抽取症状 + 城市
      → [risk_assess] risk_tool 风险分层
      → [plan] Planner 纯规则决策下一步
      → [动作节点] 生成回复/追问/升级
      → [review] 反思是否需要重规划
      → [respond] 渲染最终回复
    → memory 更新会话状态
    → RAG 检索注入知识参考
  → 返回 JSON 响应
```

## 前端架构

`templates/index.html` 单页面应用，三个 Tab：

- **对话面板**：聊天输入/消息气泡/语音按钮
- **问诊状态**：实时展示已识别症状、缺失字段、风险等级
- **配置面板**：API Key 设置 + 语音连通性测试 + 关闭入口

## 技术栈

| 组件 | 技术 |
|---|---|
| Web 框架 | Flask 2.3 |
| Agent 编排 | LangGraph 1.1 |
| HTTP 客户端 | requests |
| LLM | DeepSeek API |
| 语音 ASR | Paraformer (DashScope) |
| 语音 TTS | CosyVoice (DashScope) |
| 知识检索 | 自实现 BM25/TF-IDF（零依赖） |
| 外部工具 | MCP SDK |
| 地图服务 | 高德地图 POI API |
| 持久化 | JSON 文件（原子写入） |
| 配置 | .env 风格键值对 |
