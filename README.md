# Medical Agent

一个面向分诊与问诊场景的多轮 Agent 原型项目。

它的重点不是“自由聊天”，而是围绕一条可控主链路持续推进：

1. 结构化抽取症状和关键信息
2. 维护多轮对话状态
3. 做风险分层与动态追问
4. 生成分诊建议与问诊回复

当前默认主编排 runtime 为 `LangGraph`。

## Project Positioning

这个项目定位为：

`state-driven triage agent / 多轮分诊 Agent 原型`

它不是医疗诊断系统，重点展示的是：

- LLM 应用工程
- 状态管理
- LangGraph 编排
- 工具调用与规则控制
- 多轮对话记忆设计

## Current Architecture

项目当前可以理解成 6 层：

1. 交互层
   - [main.py](/abs/path/d:/former/研/learninglearning/medical_agent/main.py)
   - [app.py](/abs/path/d:/former/研/learninglearning/medical_agent/app.py)

2. 编排层
   - [agent/graph.py](/abs/path/d:/former/研/learninglearning/medical_agent/agent/graph.py)
   - [agent/controller.py](/abs/path/d:/former/研/learninglearning/medical_agent/agent/controller.py) `legacy fallback`
   - [agent/factory.py](/abs/path/d:/former/研/learninglearning/medical_agent/agent/factory.py)

3. 决策层
   - [agent/planner.py](/abs/path/d:/former/研/learninglearning/medical_agent/agent/planner.py)
   - [agent/router.py](/abs/path/d:/former/研/learninglearning/medical_agent/agent/router.py)

4. 记忆与存储层
   - [memory/memory.py](/abs/path/d:/former/研/learninglearning/medical_agent/memory/memory.py)
   - [memory/profile_store.py](/abs/path/d:/former/研/learninglearning/medical_agent/memory/profile_store.py)
   - [memory/session_store.py](/abs/path/d:/former/研/learninglearning/medical_agent/memory/session_store.py)

5. 工具层
   - [tools/symptom_tool.py](/abs/path/d:/former/研/learninglearning/medical_agent/tools/symptom_tool.py)
   - [tools/risk_tool.py](/abs/path/d:/former/研/learninglearning/medical_agent/tools/risk_tool.py)
   - [tools/guideline_tool.py](/abs/path/d:/former/研/learninglearning/medical_agent/tools/guideline_tool.py)

6. 模型层
   - [llm/llm.py](/abs/path/d:/former/研/learninglearning/medical_agent/llm/llm.py)
   - [llm/prompt.py](/abs/path/d:/former/研/learninglearning/medical_agent/llm/prompt.py)

## Memory Model

当前项目的记忆结构已经拆成三层：

1. `profile_store`
   - 以 `user_id` 为键
   - 保存长期用户背景
   - 当前字段包括 `age`、`sex`、`past_history`、`allergy_history`、`medication_history`

2. `session_store`
   - 以 `session_id` 为键
   - 保存当前会话的 `case_state` 和 `history`
   - 负责短期问诊状态恢复

3. `ConversationMemory`
   - 作为会话态协调器
   - 负责状态更新、摘要生成、长期画像提升、Prompt 上下文组织

这意味着当前边界已经比较清楚：

- `user_id -> profile_store`
- `session_id -> session_store`
- `graph_thread_id -> LangGraph execution thread`

在 Web 入口里，`graph_thread_id` 已经和 `session_id` 对齐。

## LangGraph Runtime

当前默认图结构为：

`extract -> risk_assess -> plan -> action subgraph -> review -> respond`

其中 `plan` 会根据 `next_action` 路由到不同 action 节点，例如：

- `risk_escalation`
- `clarify_conflict`
- `ask_followup_single`
- `ask_followup_bundle`
- `request_pulse_input`
- `final_advice`

`review` 节点支持 `replan` 回环，因此它已经不是一次性流水线，而是最小闭环 Agent。

## Core Workflow

1. 用户输入症状或补充信息
2. 系统抽取结构化字段并更新 `case_state`
3. 风险评估模块输出风险等级
4. planner 生成 `next_action`
5. LangGraph 路由到对应 action 节点
6. review 判断是否需要改判或重新规划
7. 渲染追问、高风险提示或最终建议
8. 状态回写到 `session_store / profile_store`

## Quick Start

安装依赖：

```bash
pip install -r requirements.txt
```

复制配置：

```bash
copy config.env.example config.env
```

在 `config.env` 中填写你的模型配置。

启动 Web Demo：

```bash
python app.py
```

命令行模式：

```bash
python main.py
```

运行测试：

```bash
python test_system.py
```

## Current Status

当前已经完成的关键能力：

- LangGraph 已成为默认主编排
- classic controller 降级为兼容 fallback
- action 子图已显式化
- 长期画像抽成 `profile_store`
- 会话状态抽成 `session_store`
- LangGraph `thread_id` 与业务 `session_id` 对齐

当前测试结果：

- `python test_system.py`
- `21/21 checks passed`

## Recommended Next Steps

接下来最值得做的方向：

1. 把 `checkpointer` 从内存版升级成持久化版本
2. 继续减少 `router` 在 LangGraph runtime 中的残留职责
3. 给 `guideline_tool` 接最小 RAG / retrieval 层
4. 增加 graph execution tracing / observability

## Notes

- 这是分诊与问诊辅助原型，不替代医生面诊
- 当前更强调工程结构、状态控制和 Agent 编排，而不是医学诊断结论本身
