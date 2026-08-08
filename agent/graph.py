# 生成唯一ID，每一个会话生成唯一标识
# self.thread_id = ... or str(uuid.uuid4())
# thread_id：状态隔离、恢复、保证不同用户/不同对话不混
import uuid 
# 类型注解工具
# Any:任意类型；Dict:字典类型；List:列表类型；TypedDict:类型化字典，规定这个 dict 里必须有哪些字段
# State=TypedDict(LangGraph的核心)
import time
from typing import Any, Dict, List, TypedDict
# 核心模块：规划器、路由器
# Planner:根据当前情况，决定下一步做什么
# Router:把 plan 转成具体 action
from agent.planner import Planner
from agent.router import Router
# 工具函数库
# 构建Action/动作生成：build_xxx_action_result
# *_clarify_conflict_*：有信息冲突时，问用户澄清
# *_final_advice_*：给建议
# *_followup_bundle_*：一次问多个问题
# *_followup_single_*：一次问一个问题
# *_request_pulse_input_*：请求脉搏数据
# *_risk_escalation_*：高风险下，立即就医/升级处理
# 序列化/反序列化
# serialize_action_result：把action转成可存储/传输的格式（JSON）
# deserialize_action_result：把JSON->Python对象
# 目的：langGraph需要在节点之间传递数据，所以需要可序列化
# 信息处理
# extract_case_slots：用LLM抽取结构化信息
# handle_post_pulse_reply：用户上传脉搏数据后的回复
# 响应生成
# render_response：把动作转化为自然语言
# Memory同步
# sync_action_to_memory：记录做了什么动作
# sync_plan_to_memory：记录计划细节（为什么这么做）
# sync_review_to_memory：反思结果
from agent.runtime_utils import (
    build_clarify_conflict_action_result,
    build_final_advice_action_result,
    build_followup_bundle_action_result,
    build_followup_single_action_result,
    build_request_pulse_input_action_result,
    build_risk_escalation_action_result,
    deserialize_action_result,
    extract_case_slots,
    handle_post_pulse_reply,
    render_response,
    serialize_action_result,
    sync_action_to_memory,
    sync_plan_to_memory,
    sync_review_to_memory,
)
# 外部工具（协议版）
# get_guideline_tool：根据当前情况，查相关指南/规则，返回标准 ToolResult
# risk_assessment_tool：根据当前情况，评估风险（高/中/低），返回标准 ToolResult
from observability.events import event_logger
from observability.logger import get_logger, set_trace_context
from tools.guideline_tool import get_guideline_tool
from tools.protocol import unwrap_tool_result
from tools.risk_tool import fallback_risk_result, risk_assessment_tool

_logger = get_logger("agent.graph")


# 运行风险评估工具，失败时降级为 UNKNOWN，保证问诊链路不中断
def _run_risk_assessment(case_state):
    return unwrap_tool_result(risk_assessment_tool(case_state), fallback_risk_result)


# 运行指南工具，失败时返回最小化指南
def _run_guideline(case_state, risk_result, plan=None):
    return unwrap_tool_result(
        get_guideline_tool(case_state, risk_result, plan),
        lambda: {"summary": "指南生成异常，建议谨慎参考并补充信息。", "advice": []},
    )


# 节点埋点包装：emit node_enter/node_exit 事件（含耗时），不侵入节点函数体
def _traced_node(name, func):
    def wrapper(state):
        event_logger.emit("node_enter", node=name, internal_step=state.get("internal_step", 0))
        started = time.perf_counter()
        try:
            update = func(state)
        except Exception:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            event_logger.emit("node_exit", node=name, elapsed_ms=elapsed_ms, error=True)
            raise
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        event_logger.emit("node_exit", node=name, elapsed_ms=elapsed_ms)
        return update

    return wrapper

# Agent运行时的状态结构，定义了在图中传递的数据格式
# 所有节点共享一个state
# 用户输入、对话历史
# 核心状态
# 抽取的信息
# 风险评估结果
# 计划、执行结果、反思结果等
class AgentState(TypedDict, total=False):
    user_input: str
    messages: List[Dict[str, str]]
    case_state: Dict[str, Any]
    extracted_slots: Dict[str, Any]
    risk_result: Dict[str, Any]
    plan: Dict[str, Any]
    guideline_result: Dict[str, Any]
    action_result: Any
    review_result: Dict[str, Any]
    response: str
    override_action: str
    internal_step: int

# 构建Agent的决策图
# llm：语言模型，提供理解和生成能力
# memory：存储对话历史、抽取的信息、评估结果等
# max_internal_steps：内部循环的最大次数，防止无限重试
# checkpointer：状态持久化工具，支持恢复和分析
# 返回值：编译好的图，可以直接调用invoke方法运行
def build_agent_graph(llm, memory, max_internal_steps=3, checkpointer=None):
    # 导入LangGraph核心组件
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise ImportError(
            "LangGraph is not installed. Install it first with `pip install langgraph`."
        ) from exc
    # 默认用内存版保存执行状态
    # 作用：支持中断恢复、多轮对话连续性
    if checkpointer is None:
        try:
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError as exc:
            raise ImportError(
                "LangGraph checkpointer dependencies are unavailable. "
                "Make sure `langgraph` is installed correctly."
            ) from exc
        checkpointer = InMemorySaver()

    # Planner：决定下一步做什么（纯规则决策，不依赖 LLM）
    # Router：把决策变成具体动作
    router = Router()
    planner = Planner()

    # 1.取case_state
    # 2.查医疗指南
    # 3.用router.route生成action
    def _run_routed_action(plan, risk_result):
        case_state = memory.get_case_state()
        guideline_result = _run_guideline(case_state, risk_result, plan)
        action_result = router.route(case_state, plan, risk_result, guideline_result)
        return {
            "guideline_result": guideline_result,
            "action_result": serialize_action_result(action_result),
        }

    # 定义图中的节点函数，每个函数对应一个步骤
    # def xxx_node(state: AgentState):
    # 1.读state/memory
    # 2.计算
    # 3.返回要更新的字段

    # 自然语言->结构化信息->更新case_state
    def extract_node(state: AgentState):
        user_input = state.get("user_input", "")
        memory.add_user(user_input)
        handle_post_pulse_reply(memory, user_input)

        extracted_slots = extract_case_slots(llm, user_input)
        memory.update_case(extracted_slots)

        return {
            "extracted_slots": extracted_slots,
            "case_state": memory.get_case_state(),
        }

    # 风险评估：输出风险等级（高/中/低）
    def risk_assess_node(state: AgentState):
        case_state = memory.get_case_state()
        risk_result = _run_risk_assessment(case_state)
        event_logger.emit(
            "risk_assessed",
            risk=risk_result.get("risk", ""),
            matched_rules=risk_result.get("matched_rules", []),
        )
        memory.update_triage(risk_result=risk_result)

        return {
            "risk_result": risk_result,
            "case_state": memory.get_case_state(),
        }

    # 规划：根据当前情况，决定下一步做什么
    def plan_node(state: AgentState):
        step = state.get("internal_step", 0) + 1
        case_state = memory.get_case_state()
        plan = planner.create_plan(case_state)

        override_action = state.get("override_action", "")
        if override_action:
            plan["next_action"] = override_action
            plan["action_reason"] = f"{plan.get('action_reason', '')} 自检改判：{override_action}。".strip()

        sync_plan_to_memory(memory, plan, internal_step=step)
        event_logger.emit(
            "plan",
            next_action=plan.get("next_action", ""),
            reason=(plan.get("action_reason") or "")[:120],
            internal_step=step,
        )

        return {
            "plan": plan,
            "case_state": memory.get_case_state(),
            "override_action": "",
            "internal_step": step,
        }
    # 风险升级：高风险时，直接给出建议/措施
    def risk_escalation_node(state: AgentState):
        plan = dict(state["plan"])
        plan["next_action"] = "risk_escalation"
        case_state = memory.get_case_state()
        guideline_result = _run_guideline(case_state, state["risk_result"], plan)
        action_result = build_risk_escalation_action_result(
            case_state=case_state,
            risk_result=state["risk_result"],
            guideline_result=guideline_result,
            plan=plan,
        )
        return {
            "guideline_result": guideline_result,
            "action_result": serialize_action_result(action_result),
        }

    def clarify_conflict_node(state: AgentState):
        plan = dict(state["plan"])
        plan["next_action"] = "clarify_conflict"
        case_state = memory.get_case_state()
        action_result = build_clarify_conflict_action_result(case_state=case_state, plan=plan)
        return {
            "guideline_result": {},
            "action_result": serialize_action_result(action_result),
        }

    def ask_followup_single_node(state: AgentState):
        plan = dict(state["plan"])
        plan["next_action"] = "ask_followup_single"
        case_state = memory.get_case_state()
        action_result = build_followup_single_action_result(case_state=case_state, plan=plan)
        return {
            "guideline_result": {},
            "action_result": serialize_action_result(action_result),
        }

    def ask_followup_bundle_node(state: AgentState):
        plan = dict(state["plan"])
        plan["next_action"] = "ask_followup_bundle"
        case_state = memory.get_case_state()
        action_result = build_followup_bundle_action_result(case_state=case_state, plan=plan)
        return {
            "guideline_result": {},
            "action_result": serialize_action_result(action_result),
        }

    def summarize_progress_node(state: AgentState):
        plan = dict(state["plan"])
        plan["next_action"] = "summarize_progress"
        return _run_routed_action(plan, state["risk_result"])

    def request_pulse_input_node(state: AgentState):
        plan = dict(state["plan"])
        plan["next_action"] = "request_pulse_input"
        case_state = memory.get_case_state()
        action_result = build_request_pulse_input_action_result(case_state=case_state, plan=plan)
        return {
            "guideline_result": {},
            "action_result": serialize_action_result(action_result),
        }

    def final_advice_node(state: AgentState):
        plan = dict(state["plan"])
        plan["next_action"] = "final_advice"
        case_state = memory.get_case_state()
        guideline_result = _run_guideline(case_state, state["risk_result"], plan)
        action_result = build_final_advice_action_result(
            case_state=case_state,
            risk_result=state["risk_result"],
            guideline_result=guideline_result,
            plan=plan,
        )
        return {
            "guideline_result": guideline_result,
            "action_result": serialize_action_result(action_result),
        }

    def review_node(state: AgentState):
        case_state = memory.get_case_state()
        action_result = deserialize_action_result(state["action_result"])
        review_result = planner.review_action(
            case_state=case_state,
            plan=state["plan"],
            risk_result=state["risk_result"],
            action_result=action_result,
        )
        event_logger.emit(
            "review",
            needs_replan=bool(review_result.get("needs_replan")),
            suggested_action=review_result.get("suggested_action", ""),
        )
        sync_review_to_memory(
            memory, review_result, fallback_stop_reason=state["plan"].get("stop_condition", "")
        )

        update = {
            "review_result": review_result,
            "case_state": memory.get_case_state(),
        }
        if review_result.get("needs_replan"):
            update["override_action"] = review_result.get("suggested_action", "")
        return update

    def respond_node(state: AgentState):
        case_state = memory.get_case_state()
        action_result = deserialize_action_result(state["action_result"])
        risk_result = state["risk_result"]
        guideline_result = state["guideline_result"]
        plan = state["plan"]
        response = render_response(
            llm, memory, action_result, case_state, risk_result, guideline_result, plan
        )

        sync_action_to_memory(
            memory, action_result, plan, step=state.get("internal_step", 1)
        )
        memory.add_assistant(response)

        return {
            "response": response,
            "case_state": memory.get_case_state(),
        }

    def after_review(state: AgentState):
        review_result = state.get("review_result", {})
        if review_result.get("needs_replan") and state.get("internal_step", 1) < max_internal_steps:
            return "replan"
        return "respond"

    def after_plan(state: AgentState):
        plan = state.get("plan", {})
        next_action = plan.get("next_action", "final_advice")
        if next_action not in {
            "risk_escalation",
            "clarify_conflict",
            "ask_followup_single",
            "ask_followup_bundle",
            "summarize_progress",
            "request_pulse_input",
            "final_advice",
        }:
            return "final_advice"
        return next_action

    graph = StateGraph(AgentState)
    graph.add_node("extract", _traced_node("extract", extract_node))
    graph.add_node("risk_assess", _traced_node("risk_assess", risk_assess_node))
    graph.add_node("plan", _traced_node("plan", plan_node))
    graph.add_node("risk_escalation", _traced_node("risk_escalation", risk_escalation_node))
    graph.add_node("clarify_conflict", _traced_node("clarify_conflict", clarify_conflict_node))
    graph.add_node("ask_followup_single", _traced_node("ask_followup_single", ask_followup_single_node))
    graph.add_node("ask_followup_bundle", _traced_node("ask_followup_bundle", ask_followup_bundle_node))
    graph.add_node("summarize_progress", _traced_node("summarize_progress", summarize_progress_node))
    graph.add_node("request_pulse_input", _traced_node("request_pulse_input", request_pulse_input_node))
    graph.add_node("final_advice", _traced_node("final_advice", final_advice_node))
    graph.add_node("review", _traced_node("review", review_node))
    graph.add_node("respond", _traced_node("respond", respond_node))

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "risk_assess")
    graph.add_edge("risk_assess", "plan")
    graph.add_conditional_edges(
        "plan",
        after_plan,
        {
            "risk_escalation": "risk_escalation",
            "clarify_conflict": "clarify_conflict",
            "ask_followup_single": "ask_followup_single",
            "ask_followup_bundle": "ask_followup_bundle",
            "summarize_progress": "summarize_progress",
            "request_pulse_input": "request_pulse_input",
            "final_advice": "final_advice",
        },
    )
    graph.add_edge("risk_escalation", "review")
    graph.add_edge("clarify_conflict", "review")
    graph.add_edge("ask_followup_single", "review")
    graph.add_edge("ask_followup_bundle", "review")
    graph.add_edge("summarize_progress", "review")
    graph.add_edge("request_pulse_input", "review")
    graph.add_edge("final_advice", "review")
    graph.add_conditional_edges(
        "review",
        after_review,
        {
            "replan": "plan",
            "respond": "respond",
        },
    )
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=checkpointer)

# Agent类，封装图的运行和对外接口
# 1.初始化（绑定llm\memory\graph）
# 2.提供调用接口（run）
# 3.提供系统接口（数据注入/状态查询）
class LangGraphMedicalAgent:
    def __init__(self, llm, memory, max_internal_steps=3, checkpointer=None, thread_id=None):
        self.llm = llm
        self.memory = memory
        self.max_internal_steps = max_internal_steps
        self.planner = Planner()
        self.thread_id = thread_id or getattr(memory, "session_id", None) or str(uuid.uuid4())
        self.app = build_agent_graph(
            llm=llm,
            memory=memory,
            max_internal_steps=max_internal_steps,
            checkpointer=checkpointer,
        )

    def _invoke_config(self):
        return {"configurable": {"thread_id": self.thread_id}}

    def run(self, user_input):
        # 每轮生成 turn_id，串联本次提问的全链路事件
        turn_id = uuid.uuid4().hex[:12]
        session_id = getattr(self.memory, "session_id", "") or self.thread_id
        set_trace_context(session_id=session_id, turn_id=turn_id)

        started = time.perf_counter()
        try:
            result = self.app.invoke(
                {
                    "user_input": user_input,
                    "messages": [{"role": "user", "content": user_input}],
                    "case_state": self.memory.get_case_state(),
                    "override_action": "",
                    "internal_step": 0,
                },
                self._invoke_config(),
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            event_logger.emit("run_end", turn_id=turn_id, error=str(exc), elapsed_ms=elapsed_ms)
            _logger.error("Agent run failed turn_id=%s: %s", turn_id, exc)
            raise

        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        event_logger.emit(
            "run_end",
            turn_id=turn_id,
            elapsed_ms=elapsed_ms,
            internal_step=result.get("internal_step", 1),
        )
        _logger.info("Agent run finished turn_id=%s elapsed_ms=%s", turn_id, elapsed_ms)
        return result.get("response", "")

    # 提供系统接口，允许外部注入数据（如脉搏数据），并触发图的相关节点更新状态
    # 作用：动态更新状态、触发后续步骤、支持多轮交互
    def ingest_pulse_data(self, pulse_data):
        self.memory.update_pulse_data(pulse_data)
        case_state = self.memory.get_case_state()
        risk_result = _run_risk_assessment(case_state)
        plan = self.planner.create_plan(case_state)
        self.memory.update_triage(risk_result=risk_result)
        sync_plan_to_memory(self.memory, plan)
        return self.get_case_snapshot()

    # 提供系统接口，允许外部查询当前状态快照
    # 包括case_state、长期画像、LLM状态等，方便监控和分析
    # 作用：状态可视化、调试、用户反馈等
    def get_case_snapshot(self):
        snapshot = self.memory.get_case_state()
        snapshot["long_term_profile"] = self.memory.get_long_term_profile()
        snapshot["llm_status"] = self.llm.get_runtime_status()
        snapshot["session_id"] = getattr(self.memory, "session_id", "")
        snapshot["user_id"] = getattr(self.memory, "user_id", "")
        snapshot["graph_thread_id"] = self.thread_id
        return snapshot

    # 提供系统接口，允许外部查询当前图的完整状态，包括所有节点的输入输出，支持调试和分析
    # 作用：图状态可视化、调试、性能分析等
    def get_graph_state(self):
        return self.app.get_state(self._invoke_config())
