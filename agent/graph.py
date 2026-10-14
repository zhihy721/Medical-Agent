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
from agent.safety_gate import assert_response_action
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
    red_flag_correction_notice,
    resolve_clarified_contradiction,
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
    if 'fact_ledger' in case_state:
        from memory.fact_ledger_runtime import risk_input
        case_state = risk_input(case_state['fact_ledger'])
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
    override_plan: dict
    decision_trace: list
    extraction_trace: dict
    defer_hospital: bool

# 构建Agent的决策图
# llm：语言模型，提供理解和生成能力
# memory：存储对话历史、抽取的信息、评估结果等
# max_internal_steps：内部循环的最大次数，防止无限重试
# checkpointer：状态持久化工具，支持恢复和分析
# 返回值：编译好的图，可以直接调用invoke方法运行
def build_agent_graph(llm, memory, max_internal_steps=3, checkpointer=None, decision_policy=None,
                      fact_extractor=None, responder=None, response_mode='legacy'):
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
        if getattr(fact_extractor,'turn_requests_version',None)=='joint_interaction_v1':
            from agent.interaction_intent import begin
            memory.case_state['interaction_sequence']=memory.case_state.get('interaction_sequence',0)+1
            begin(memory,user_input)
        if getattr(fact_extractor,'turn_requests_version',None)=='delivery_turn_requests_v1':
            from agent.turn_requests import update
            update(memory,user_input)

        if fact_extractor is not None:
            audit = fact_extractor.run(memory, user_input)
            if getattr(fact_extractor,'turn_requests_version',None)=='delivery_turn_requests_v1':
                from agent.delivery_recovery import build
                memory.case_state['delivery_recovery']=build(memory.get_case_state(),user_input)
            return {'extracted_slots': {}, 'case_state': memory.get_case_state(), 'extraction_trace': audit}
        extracted_slots = extract_case_slots(llm, user_input)
        memory.update_case(extracted_slots)
        # 澄清回复后处理：用户重新确认冲突字段时以最新值为准消除矛盾
        resolve_clarified_contradiction(memory, extracted_slots)

        return {
            "extracted_slots": extracted_slots,
            "case_state": memory.get_case_state(),
        }

    # 风险评估：输出风险等级（高/中/低）
    def risk_assess_node(state: AgentState):
        case_state = memory.get_case_state()
        risk_result = _run_risk_assessment(case_state)
        safety_version=getattr(fact_extractor,'input_safety_version',None)
        if safety_version:
            if safety_version=='literal_breathing_safety_v1':
                from agent.input_safety_signal_v1 import apply,VERSION
            else:
                from agent.input_safety_signal import apply,VERSION
            if safety_version!=VERSION:raise ValueError('unsupported input safety version')
            risk_result=apply(memory,state.get('user_input',''),risk_result)
        event_logger.emit(
            "risk_assessed",
            risk=risk_result.get("risk", ""),
            matched_rules=risk_result.get("matched_rules", []),
        )
        memory.update_triage(risk_result=risk_result)
        intent = getattr(fact_extractor, 'intent', None)
        if intent is not None:
            if risk_result.get('risk') != 'HIGH' and (not case_state.get('fact_extraction_error') or getattr(intent,'joint_interaction',False)):
                intent.run(memory, state.get('user_input', ''))
            else:
                memory.apply_complaint_intent({'status': 'skipped', 'events': [],
                    'reason': 'high_risk_or_fact_failure'})

        return {
            "risk_result": risk_result,
            "case_state": memory.get_case_state(),
        }

    # 规划：根据当前情况，决定下一步做什么
    def plan_node(state: AgentState):
        step = state.get("internal_step", 0) + 1
        case_state = memory.get_case_state()
        decision_trace = list(state.get("decision_trace") or [])
        if getattr(decision_policy,'interaction_policy',False):
            plan,audit=decision_policy.plan(case_state,state['risk_result'])
            decision_trace.append(audit)
            plan,review_audit=decision_policy.review_plan(case_state,plan,state['risk_result'])
            decision_trace.append(review_audit)
        elif case_state.get('fact_extraction_error') and state['risk_result'].get('risk') != 'HIGH':
            plan = planner.create_plan(case_state)
            question=(case_state.get('delivery_recovery') or {}).get('question') or '本轮信息未能可靠解析，请重新描述。'
            plan.update(next_action='clarify_conflict', action_reason='本轮信息解析失败，需要重新描述。',
                        contradictions=[question],
                        question_targets=[],question_target_fields=[],pending_questions=[])
        elif decision_policy is not None:
            plan, audit = decision_policy.plan(case_state, state["risk_result"], state.get("override_plan"))
            decision_trace.append(audit)
            event_logger.emit("decision_gate", stage="plan", **audit["gate"])
        else:
            plan = planner.create_plan(case_state)
            override_action = state.get("override_action", "")
            if override_action:
                plan["next_action"] = override_action
                plan["action_reason"] = f"{plan.get('action_reason', '')} 自检改判：{override_action}。".strip()

        request=case_state.get('delivery_turn_request') or {}
        if (request.get('version')=='delivery_turn_requests_v1' and request.get('summary')
                and not case_state.get('fact_extraction_error') and state['risk_result'].get('risk')!='HIGH'):
            from agent.turn_requests import summary_plan
            plan=summary_plan(plan,request)
        elif (state['risk_result'].get('risk')!='HIGH' and not case_state.get('fact_extraction_error')
              and not case_state.get('contradictions') and (case_state.get('delivery_recovery') or {}).get('kind')=='missing_deferred_candidate'):
            question=case_state['delivery_recovery']['question']
            plan.update(next_action='ask_followup_single',next_focus='input_clarification',
                        question_targets=[],question_target_fields=[],pending_questions=[question],
                        input_clarification=case_state['delivery_recovery'],action_reason='当前症状候选遗漏，先确认是否确实发生。')
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
            "override_plan": {},
            "decision_trace": decision_trace,
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
            include_hospitals=not state.get('defer_hospital',False) and not case_state.get('interaction_turn',{}).get('tools_forbidden'),
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
        if plan.get('input_clarification'):
            from agent.router import ActionResult
            action_result=ActionResult('ask_followup_single',plan['input_clarification']['question'],'COLLECTING_INFO',False,[],render_mode='followup')
        else:action_result = build_followup_single_action_result(case_state=case_state, plan=plan)
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
        if plan.get('requested_progress'):
            from agent.runtime_utils import build_requested_progress_action_result
            return {'guideline_result':{},'action_result':serialize_action_result(
                build_requested_progress_action_result(memory,plan))}
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
        decision_trace = list(state.get("decision_trace") or [])
        override_plan = None
        if getattr(decision_policy,'pre_execute_review',False):
            review_result={'needs_replan':False,'suggested_action':'','reason':'已在工具执行前审查；执行后不重放工具。'}
        elif case_state.get('fact_extraction_error'):
            review_result = {'needs_replan': False, 'suggested_action': '',
                             'reason': '保留已知风险处置或请求重述，未消费失败输出。'}
        elif decision_policy is not None:
            review_result, override_plan, audit = decision_policy.review(case_state, state["plan"], state["risk_result"], action_result)
            if review_result.get("needs_replan") and state.get("internal_step", 1) >= max_internal_steps:
                audit["replan_exhausted"] = True
                review_result["replan_exhausted"] = True
            decision_trace.append(audit)
            if audit.get("gate"):
                event_logger.emit("decision_gate", stage="review", **audit["gate"])
        else:
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
            "decision_trace": decision_trace,
            "override_plan": override_plan or {},
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
        if decision_policy is not None:
            assert_response_action(action_result.name, risk_result, case_state)
        response = render_response(
            llm, memory, action_result, case_state, risk_result, guideline_result, plan,
            responder=responder, response_mode=response_mode,
        )
        response = red_flag_correction_notice(case_state) + response
        if (case_state.get('fact_extraction_error') and risk_result.get('risk') != 'HIGH'
                and not getattr(decision_policy,'interaction_policy',False)):
            response = (case_state.get('delivery_recovery') or {}).get('question') or '本轮信息未能可靠识别，尚未更新症状记录。请重新描述这次不适；如有紧急不适，请及时线下就医。'
        if plan.get('input_clarification'):
            response=plan['input_clarification']['question']
            key=plan['input_clarification']['key']
            counts=dict(case_state.get('delivery_clarification_counts',{}));counts[key]=counts.get(key,0)+1
            memory.case_state['delivery_clarification_counts']=counts

        # This is an adjunct to the already selected safe action, never a
        # replacement for HIGH disposition or an invitation to wait for help.
        clarification = None
        counts = dict(case_state.get('interpretation_question_counts', {}))
        from agent.qwen_interaction_policy import paused
        if not case_state.get('fact_extraction_error') and not (getattr(decision_policy,'interaction_policy',False) and paused(case_state)):
            clarification = next((item for item in case_state.get('interpretation_annotations', [])
                if item['interpretation_status'] == 'needs_clarification'
                and counts.get(item['fact_id'], 0) < 2), None)
        if clarification:
            response += '\n\n请先按上面的安全建议处理，不要等待问答。若能够安全回复：' + clarification['question']
            counts[clarification['fact_id']] = counts.get(clarification['fact_id'], 0) + 1
            memory.case_state['interpretation_question_counts'] = counts

        confirmation = plan.get('fact_confirmation')
        if confirmation and action_result.name == 'ask_followup_single':
            # Preserve the deterministic presence question if a responder rewrites it.
            response = action_result.response
            confirmation_counts = dict(case_state.get('fact_confirmation_counts', {}))
            fid = confirmation['fact_id']
            confirmation_counts[fid] = confirmation_counts.get(fid, 0) + 1
            memory.case_state['fact_confirmation_counts'] = confirmation_counts

        sync_action_to_memory(
            memory, action_result, plan, step=state.get("internal_step", 1)
        )
        memory.add_assistant(response)
        if fact_extractor is not None:
            memory.set_fact_question(clarification['question'] if clarification else response if action_result.name in {
                'ask_followup_single', 'ask_followup_bundle', 'clarify_conflict'} else '')

        return {
            "response": response,
            "case_state": memory.get_case_state(),
        }

    def after_review(state: AgentState):
        review_result = state.get("review_result", {})
        if review_result.get("needs_replan") and state.get("internal_step", 1) < max_internal_steps:
            return "replan"
        return "respond"

    def hospital_enrichment_node(state: AgentState):
        # In streaming mode this node cannot begin until the consumer resumes
        # after receiving the preceding respond update. No detached worker.
        from agent.runtime_utils import _nearby_hospital_lines
        lines = [] if memory.case_state.get('interaction_turn',{}).get('tools_forbidden') else _nearby_hospital_lines(memory.get_case_state())
        response = state['response']
        if lines:
            enriched = response + '\n' + '\n'.join(lines)
            memory.amend_last_assistant(response, enriched)
            response = enriched
        action = dict(state['action_result'])
        action['response'] = response
        return {'response':response,'action_result':action,'case_state':memory.get_case_state()}

    def after_respond(state: AgentState):
        if state.get('defer_hospital') and state['risk_result'].get('risk') == 'HIGH':
            return 'hospital_enrichment'
        return 'end'

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
            "acknowledge_pause", "clarify_interaction_request", "provide_hospital_options",
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
    def interaction_node(state):
        from agent.interaction_actions import build
        result=build(memory,state['plan'],state['risk_result'])
        return {'guideline_result':{},'action_result':serialize_action_result(result)}
    for name in ('acknowledge_pause','clarify_interaction_request','provide_hospital_options'):
        graph.add_node(name,_traced_node(name,interaction_node))
    graph.add_node("review", _traced_node("review", review_node))
    graph.add_node("respond", _traced_node("respond", respond_node))
    graph.add_node('hospital_enrichment', _traced_node('hospital_enrichment',hospital_enrichment_node))

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
            "acknowledge_pause":"acknowledge_pause", "clarify_interaction_request":"clarify_interaction_request",
            "provide_hospital_options":"provide_hospital_options",
        },
    )
    graph.add_edge("risk_escalation", "review")
    graph.add_edge("clarify_conflict", "review")
    graph.add_edge("ask_followup_single", "review")
    graph.add_edge("ask_followup_bundle", "review")
    graph.add_edge("summarize_progress", "review")
    graph.add_edge("request_pulse_input", "review")
    graph.add_edge("final_advice", "review")
    for name in ('acknowledge_pause','clarify_interaction_request','provide_hospital_options'):graph.add_edge(name,'review')
    graph.add_conditional_edges(
        "review",
        after_review,
        {
            "replan": "plan",
            "respond": "respond",
        },
    )
    graph.add_conditional_edges('respond',after_respond,{'hospital_enrichment':'hospital_enrichment','end':END})
    graph.add_edge('hospital_enrichment',END)

    return graph.compile(checkpointer=checkpointer)

# Agent类，封装图的运行和对外接口
# 1.初始化（绑定llm\memory\graph）
# 2.提供调用接口（run）
# 3.提供系统接口（数据注入/状态查询）
class LangGraphMedicalAgent:
    def __init__(self, llm, memory, max_internal_steps=3, checkpointer=None, thread_id=None, decision_policy=None,
                 fact_extractor=None, responder=None, response_mode=None):
        self.llm = llm
        self.memory = memory
        self.max_internal_steps = max_internal_steps
        self.planner = Planner()
        self.decision_policy = decision_policy
        if fact_extractor is not None:
            memory.enable_fact_mode(getattr(fact_extractor, 'contract_version', None),
                                    representation_policy=getattr(fact_extractor, 'representation_policy', None))
            projection = getattr(fact_extractor, 'runtime_projection_version', None)
            if projection:
                memory.enable_runtime_fact_projection(projection)
        elif memory.get_case_state().get('semantic_contract'):
            raise ValueError('restored semantic session requires semantic extractor')
        self.fact_extractor = fact_extractor
        self.response_mode = response_mode or ('template' if fact_extractor else 'legacy')
        if self.response_mode not in {'legacy', 'template', 'model'}:
            raise ValueError('unsupported response mode')
        if fact_extractor is not None and self.response_mode == 'legacy':
            raise ValueError('semantic sessions require explicit template/model response mode')
        if self.response_mode == 'model' and responder is None:
            raise ValueError('model response mode requires independent responder client')
        self.thread_id = thread_id or getattr(memory, "session_id", None) or str(uuid.uuid4())
        self.app = build_agent_graph(
            llm=llm,
            memory=memory,
            max_internal_steps=max_internal_steps,
            checkpointer=checkpointer,
            decision_policy=decision_policy,
            fact_extractor=fact_extractor, responder=responder, response_mode=self.response_mode,
        )

    def _invoke_config(self):
        return {"configurable": {"thread_id": self.thread_id}}

    def run(self, user_input, user_coords=None):
        if user_coords:
            self.memory.update_user_coords(user_coords["latitude"], user_coords["longitude"])
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
                    "override_plan": {},
                    "decision_trace": [],
                    "defer_hospital": False,
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

    def run_stream(self, user_input, user_coords=None):
        """Yield a HIGH safety reply before optional hospital lookup; then final.

        Caller must consume/close under its session lock. This proves delivery
        order at the generator boundary, not clinical correctness or rendering
        in a disconnected client's browser.
        """
        if user_coords:
            self.memory.update_user_coords(user_coords['latitude'],user_coords['longitude'])
        turn_id=uuid.uuid4().hex[:12]
        set_trace_context(session_id=getattr(self.memory,'session_id','') or self.thread_id,turn_id=turn_id)
        started=time.perf_counter();completed=False;error=''
        stream=self.app.stream({'user_input':user_input,'messages':[{'role':'user','content':user_input}],
            'case_state':self.memory.get_case_state(),'override_action':'','internal_step':0,
            'override_plan':{},'decision_trace':[],'defer_hospital':True},
            self._invoke_config(),stream_mode='updates')
        try:
            for update in stream:
                response=update.get('respond')
                if response and response['case_state'].get('risk_level')=='HIGH':
                    event_logger.emit('safety_notice_yielded',turn_id=turn_id)
                    yield {'kind':'safety_notice','response':response['response'],'turn_id':turn_id}
            completed=True
            state=self.get_graph_state().values
            yield {'kind':'final','response':state.get('response',''),'turn_id':turn_id}
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc)
            raise
        finally:
            stream.close()
            event_logger.emit('run_end',turn_id=turn_id,elapsed_ms=round((time.perf_counter()-started)*1000,1),
                streamed=True,completed=completed,cancelled=not completed and not error,error=error)

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
        if self.decision_policy is not None:
            snapshot["decision_policy"] = self.decision_policy.summary()
            state = self.get_graph_state().values
            snapshot["decision_audit"] = [
                {"stage": item.get("stage"), "source": item.get("source"), "gate": item.get("gate"),
                 "fallback_used": item.get("fallback_used", False)}
                for item in state.get("decision_trace", [])
            ]
        return snapshot

    # 提供系统接口，允许外部查询当前图的完整状态，包括所有节点的输入输出，支持调试和分析
    # 作用：图状态可视化、调试、性能分析等
    def get_graph_state(self):
        return self.app.get_state(self._invoke_config())
