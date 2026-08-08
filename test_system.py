#!/usr/bin/env python3

import sys
import os

# 检查langgraph是否安装，如果安装了就优先使用langgraph运行时
# 用create_agent接口创建一个langgraph agent进行测试
def _create_langgraph_agent():
    from agent.factory import create_agent, is_langgraph_available
    from llm.llm import LLM
    from llm.prompt import SYSTEM_PROMPT
    from memory.memory import ConversationMemory

    if not is_langgraph_available():
        print("[SKIP] langgraph tests (langgraph not installed)")
        return None

    return create_agent(
        llm=LLM(system_prompt=SYSTEM_PROMPT),
        memory=ConversationMemory(),
        runtime="langgraph",
    )

# 测试默认运行时是否优先使用langgraph（如果安装了的话）
def test_default_runtime_prefers_langgraph():
    from agent.factory import get_agent_runtime

    previous = os.environ.pop("AGENT_RUNTIME", None)
    try:
        runtime = get_agent_runtime()
    finally:
        if previous is not None:
            os.environ["AGENT_RUNTIME"] = previous

    if runtime == "langgraph":
        print("[PASS] default runtime prefers langgraph")
        return True

    print("[FAIL] default runtime prefers langgraph")
    return False

# 测试所有模块是否能成功导入，确保没有语法错误或缺失依赖
def test_imports():
    modules = [
        "agent.controller",
        "agent.router",
        "agent.planner",
        "llm.llm",
        "llm.prompt",
        "tools.symptom_tool",
        "tools.risk_tool",
        "tools.guideline_tool",
        "memory.memory",
        "memory.profile_store",
        "memory.session_store",
        "knowledge.tcm_knowledge",
        "agent.graph",
        "agent.factory",
        "tools.protocol",
        "tools.registry",
        "observability.logger",
        "observability.events",
        "observability.metrics",
    ]

    passed = True
    for module in modules:
        try:
            __import__(module)
            print(f"[PASS] import {module}")
        except Exception as exc:
            passed = False
            print(f"[FAIL] import {module}: {exc}")
    return passed

# 测试一个完整的问诊流程，验证症状提取、状态更新和响应生成等核心功能
def test_agent_flow():
    from agent.controller import MedicalAgent
    from llm.llm import LLM
    from llm.prompt import SYSTEM_PROMPT
    from memory.memory import ConversationMemory

    agent = MedicalAgent(LLM(system_prompt=SYSTEM_PROMPT), ConversationMemory())
    response_1 = agent.run("我发烧两天了，还有咳嗽")
    response_2 = agent.run("中度，主要在咽喉和胸口")
    case_state = agent.get_case_snapshot()

    checks = [
        ("response 1 is not empty", bool(response_1.strip())),
        ("response 2 is not empty", bool(response_2.strip())),
        ("symptoms extracted", bool(case_state["symptoms"])),
        ("status recorded", bool(case_state["status"])),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试高危症状是否能正确触发风险升级和紧急处理流程，并在响应中体现相关建议或处置信息
def test_high_risk_escalation():
    from agent.controller import MedicalAgent
    from llm.llm import LLM
    from llm.prompt import SYSTEM_PROMPT
    from memory.memory import ConversationMemory

    agent = MedicalAgent(LLM(system_prompt=SYSTEM_PROMPT), ConversationMemory())
    response = agent.run("我胸痛，还喘不过气，很严重")
    case_state = agent.get_case_snapshot()

    checks = [
        ("high risk status recorded", case_state["risk_level"] == "HIGH"),
        ("emergency status recorded", case_state["status"] == "EMERGENCY_ESCALATION"),
        ("response mentions disposition", "建议" in response or "急诊" in response),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试跟进策略是否能根据缺失信息和用户输入动态调整提问策略，并在响应中体现针对性的追问或建议
def test_followup_strategy():
    from agent.controller import MedicalAgent
    from llm.llm import LLM
    from llm.prompt import SYSTEM_PROMPT
    from memory.memory import ConversationMemory

    agent = MedicalAgent(LLM(system_prompt=SYSTEM_PROMPT), ConversationMemory())
    response = agent.run("我发热，还咳嗽")
    case_state = agent.get_case_snapshot()

    checks = [
        ("collecting info status recorded", case_state["status"] == "COLLECTING_INFO"),
        ("followup remembers missing slots", len(case_state["missing_slots"]) >= 1),
        ("response asks for more detail", "补充" in response or "持续" in response or "严重程度" in response),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试中医证候信息和脉诊数据是否能正确提取、存储并影响后续的诊断推理和响应生成
def test_tcm_evidence_and_pulse():
    from agent.controller import MedicalAgent
    from llm.llm import LLM
    from llm.prompt import SYSTEM_PROMPT
    from memory.memory import ConversationMemory

    agent = MedicalAgent(LLM(system_prompt=SYSTEM_PROMPT), ConversationMemory())
    agent.run("我腹痛，按着舒服，怕冷，没胃口，大便稀")
    snapshot = agent.ingest_pulse_data(
        {
            "pulse_summary": "细弱",
            "pulse_signal_quality": "高",
            "pulse_candidates": ["细弱", "沉"],
        }
    )
    case_state = snapshot["case_state"] if "case_state" in snapshot else snapshot

    checks = [
        ("tcm cold_heat extracted", bool(case_state["cold_heat"])),
        ("tcm appetite extracted", bool(case_state["appetite"])),
        ("pulse summary stored", case_state["pulse_summary"] == "细弱"),
        ("syndrome candidates updated", bool(case_state["syndrome_candidates"])),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试可选信息项在用户未提供时不会重复提问
# 但会被记录为已缺失并在后续诊断中考虑其缺失状态
def test_optional_slot_not_repeated():
    from agent.planner import Planner
    from llm.llm import LLM
    from llm.prompt import SYSTEM_PROMPT

    planner = Planner(LLM(system_prompt=SYSTEM_PROMPT))
    case_state = {
        "chief_complaint": "腹痛",
        "symptoms": ["腹痛"],
        "duration": "两天",
        "severity": "中度",
        "location": "腹部",
        "cold_heat": "怕冷",
        "appetite": "纳差",
        "sleep": "",
        "stool_urine": "",
        "pain_character": "",
        "followup_counts": {"stool_urine": 1},
        "risk_level": "LOW",
        "summary": "症状: 腹痛；病程: 两天；程度: 中度",
        "tcm_summary": "寒热: 怕冷；纳食: 纳差",
        "accompanying_symptoms": [],
    }
    plan = planner.create_plan(case_state)

    checks = [
        ("stool slot deferred after one miss", "stool_urine" not in plan["missing_slots"]),
        ("deferred slots record stool", "stool_urine" in plan["deferred_slots"]),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试当用户明确表示无法提供脉诊信息时，系统能够正确记录该信息为已跳过，并且不再重复询问
# 同时在后续诊断推理中考虑脉诊信息的缺失状态
def test_pulse_prompt_does_not_repeat():
    from agent.controller import MedicalAgent
    from llm.llm import LLM
    from llm.prompt import SYSTEM_PROMPT
    from memory.memory import ConversationMemory

    memory = ConversationMemory()
    memory.update_case(
        {
            "chief_complaint": "腹痛",
            "symptoms": ["腹痛"],
            "duration": "三天",
            "severity": "中度",
            "location": "腹部",
            "cold_heat": "怕冷",
            "appetite": "食欲正常",
            "sleep": "失眠",
            "stool_urine": "便秘",
        }
    )
    memory.update_triage(last_action="request_pulse_input", status="WAITING_PULSE_INPUT")
    agent = MedicalAgent(LLM(system_prompt=SYSTEM_PROMPT), memory)
    second = agent.run("没有脉诊设备，直接分析吧")
    case_state = agent.get_case_snapshot()

    checks = [
        ("second response moves forward", case_state["status"] != "WAITING_PULSE_INPUT"),
        ("pulse marked skipped", case_state["pulse_declined"] is True),
        ("second response not asking pulse again", "如果你手边有脉诊设备" not in second),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试长期画像的更新和提升机制，验证当新的病史信息被添加后
# 长期画像能够正确提取并提升相关信息，同时当前病例状态仍保留急性症状等信息
def test_long_term_profile_promotion():
    from memory.memory import ConversationMemory

    memory = ConversationMemory()
    memory.update_case(
        {
            "age": "45",
            "sex": "女",
            "past_history": ["高血压", "糖尿病"],
            "allergy_history": ["药物过敏"],
            "medication_history": ["阿司匹林"],
            "symptoms": ["咳嗽"],
            "duration": "三天",
        }
    )
    profile = memory.get_long_term_profile()
    case_state = memory.get_case_state()

    checks = [
        ("long-term age promoted", profile["age"] == "45"),
        ("long-term sex promoted", profile["sex"] == "女"),
        ("long-term past history promoted", "高血压" in profile["past_history"] and "糖尿病" in profile["past_history"]),
        ("long-term allergy promoted", "药物过敏" in profile["allergy_history"]),
        ("long-term medication promoted", "阿司匹林" in profile["medication_history"]),
        ("current case state still keeps acute info", "咳嗽" in case_state["symptoms"]),
        ("profile summary generated", profile["profile_summary"] != "长期画像暂未形成"),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试当同一用户在不同会话中使用系统时，长期画像能够被正确复用
# 不会丢失之前会话中记录的病史信息
def test_profile_store_reuses_long_term_profile_across_sessions():
    from memory.memory import ConversationMemory
    from memory.profile_store import InMemoryProfileStore

    profile_store = InMemoryProfileStore()
    first_memory = ConversationMemory(profile_store=profile_store, user_id="shared-user")
    first_memory.update_case(
        {
            "age": "52",
            "sex": "男",
            "past_history": ["高血压"],
            "medication_history": ["降压药"],
            "symptoms": ["头痛"],
        }
    )

    second_memory = ConversationMemory(profile_store=profile_store, user_id="shared-user")
    reused_profile = second_memory.get_long_term_profile()
    second_case_state = second_memory.get_case_state()

    checks = [
        ("profile store reuses age for same user", reused_profile["age"] == "52"),
        ("profile store reuses sex for same user", reused_profile["sex"] == "男"),
        ("profile store reuses history for same user", "高血压" in reused_profile["past_history"]),
        ("profile store reuses medication for same user", "降压药" in reused_profile["medication_history"]),
        ("new session does not inherit acute symptoms", not second_case_state["symptoms"]),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试当同一会话中创建多个内存实例时，是否能够正确复用会话状态
# 确保之前内存实例中记录的病史信息和状态更新能够
def test_session_store_reuses_session_state_across_memory_instances():
    from memory.memory import ConversationMemory
    from memory.session_store import InMemorySessionStore

    session_store = InMemorySessionStore()
    first_memory = ConversationMemory(session_store=session_store, session_id="shared-session")
    first_memory.add_user("我头痛三天")
    first_memory.update_case(
        {
            "chief_complaint": "头痛",
            "symptoms": ["头痛"],
            "duration": "三天",
        }
    )

    second_memory = ConversationMemory(session_store=session_store, session_id="shared-session")
    restored_case_state = second_memory.get_case_state()
    restored_history = second_memory.get_context()

    checks = [
        ("session store restores chief complaint", restored_case_state["chief_complaint"] == "头痛"),
        ("session store restores symptoms", "头痛" in restored_case_state["symptoms"]),
        ("session store restores duration", restored_case_state["duration"] == "三天"),
        ("session store restores recent history", bool(restored_history) and restored_history[0][0] == "user"),
        ("session store does not depend on profile reuse", restored_case_state["status"] == "INIT"),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试对话总结和未决问题的生成逻辑，验证当用户提供了新的症状信息后
# 系统能够正确总结当前对话内容，提取已确认的症状信息，并生成针对性的未决问题列表
def test_conversation_summary_and_open_questions():
    from memory.memory import ConversationMemory

    memory = ConversationMemory()
    memory.add_user("我发热两天了，还有咳嗽")
    memory.update_case(
        {
            "chief_complaint": "发热",
            "symptoms": ["发热", "咳嗽"],
            "duration": "两天",
            "severity": "中度",
        }
    )
    memory.update_task_state(
        goal_progress="正在补充3项关键四诊信息",
        pending_questions=["寒热倾向", "食欲与纳食", "睡眠情况"],
    )
    memory.add_assistant("我还想了解一下寒热、食欲和睡眠情况。")

    case_state = memory.get_case_state()
    prompt_context = memory.get_prompt_context_text()

    checks = [
        ("open questions mirrored", case_state["open_questions"] == ["寒热倾向", "食欲与纳食", "睡眠情况"]),
        ("resolved facts generated", bool(case_state["resolved_facts"])),
        ("conversation summary generated", "已确认" in case_state["conversation_summary"]),
        ("prompt context includes summary", "summary:" in prompt_context),
        ("prompt context keeps recent history", "recent_history:" in prompt_context),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试使用langgraph运行时时，是否能够成功执行一个简单的问诊流程
# 并且在图状态中正确记录用户输入和病例状态等信息
def test_langgraph_runtime_smoke():
    agent = _create_langgraph_agent()
    if agent is None:
        return True
    response = agent.run("我发热两天了，还有咳嗽")
    case_state = agent.get_case_snapshot()

    checks = [
        ("langgraph response is not empty", bool(response.strip())),
        ("langgraph runtime name recorded", getattr(agent, "runtime_name", "") == "langgraph"),
        ("langgraph case state updated", bool(case_state["status"])),
        ("langgraph stores symptoms", bool(case_state["symptoms"])),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试当使用langgraph运行时时，高危症状是否能正确触发风险升级和紧急处理流程
# 并在响应中体现相关建议或处置信息
def test_langgraph_high_risk_escalation():
    agent = _create_langgraph_agent()
    if agent is None:
        return True

    response = agent.run("我胸痛，还喘不过气，很严重")
    case_state = agent.get_case_snapshot()

    checks = [
        ("langgraph high risk level recorded", case_state["risk_level"] == "HIGH"),
        ("langgraph emergency status recorded", case_state["status"] == "EMERGENCY_ESCALATION"),
        ("langgraph emergency action recorded", case_state["last_action"] == "risk_escalation"),
        ("langgraph emergency response mentions disposition", "建议" in response or "急诊" in response),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试当使用langgraph运行时时，系统在请求用户提供脉诊信息后
# 如果用户明确表示无法提供，系统能够正确记录该信息为已跳过，并且不再重复询问
# 同时在后续诊断推理中考虑脉诊信息的缺失状态
def test_langgraph_pulse_prompt_does_not_repeat():
    from agent.factory import create_agent, is_langgraph_available
    from llm.llm import LLM
    from llm.prompt import SYSTEM_PROMPT
    from memory.memory import ConversationMemory

    if not is_langgraph_available():
        print("[SKIP] langgraph pulse branch (langgraph not installed)")
        return True

    memory = ConversationMemory()
    memory.update_case(
        {
            "chief_complaint": "腹痛",
            "symptoms": ["腹痛"],
            "duration": "三天",
            "severity": "中度",
            "location": "腹部",
            "cold_heat": "怕冷",
            "appetite": "食欲正常",
            "sleep": "失眠",
            "stool_urine": "便秘",
        }
    )
    memory.update_triage(last_action="request_pulse_input", status="WAITING_PULSE_INPUT")
    agent = create_agent(
        llm=LLM(system_prompt=SYSTEM_PROMPT),
        memory=memory,
        runtime="langgraph",
    )
    response = agent.run("没有脉诊设备，直接分析吧")
    case_state = agent.get_case_snapshot()

    checks = [
        ("langgraph pulse flow moves forward", case_state["status"] != "WAITING_PULSE_INPUT"),
        ("langgraph pulse marked skipped", case_state["pulse_declined"] is True),
        ("langgraph pulse prompt not repeated", "如果你手边有脉诊设备" not in response),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试当使用langgraph运行时时，系统在图状态中正确记录线程ID等信息
# 确保线程ID与会话ID一致，并且图状态中能够正确
# 记录用户输入和病例状态等信息，验证langgraph的checkpointer功能正常工作
def test_langgraph_checkpointer_thread_state():
    agent = _create_langgraph_agent()
    if agent is None:
        return True

    agent.run("我发热两天了，还有咳嗽")
    graph_state = agent.get_graph_state()
    snapshot = agent.get_case_snapshot()
    values = getattr(graph_state, "values", {})

    checks = [
        ("langgraph thread id exposed", bool(snapshot.get("graph_thread_id"))),
        ("langgraph thread id aligns with session id", snapshot.get("graph_thread_id") == snapshot.get("session_id")),
        ("langgraph graph state has values", isinstance(values, dict) and bool(values)),
        ("langgraph graph state keeps user input", values.get("user_input") == "我发热两天了，还有咳嗽"),
        ("langgraph graph state keeps case state", isinstance(values.get("case_state"), dict)),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试当使用langgraph运行时时，系统在图状态中正确记录和处理action subgraph的输出
# 验证当用户输入高危症状时，langgraph能够正确触发
# risk_escalation action subgraph，并在图状态中记录相关的action result信息
def test_langgraph_action_subgraph_routes():
    agent = _create_langgraph_agent()
    if agent is None:
        return True

    agent.run("我胸痛，还喘不过气，很严重")
    graph_state = agent.get_graph_state()
    values = getattr(graph_state, "values", {})
    action_payload = values.get("action_result", {})

    checks = [
        ("langgraph action result serialized as dict", isinstance(action_payload, dict)),
        ("langgraph action route resolved to risk escalation", action_payload.get("name") == "risk_escalation"),
        ("langgraph graph state keeps plan", isinstance(values.get("plan"), dict)),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试当使用langgraph运行时时
# 系统在图状态中正确记录和处理请求用户提供信息的action subgraph输出
def test_langgraph_pulse_action_subgraph_route():
    from agent.runtime_utils import build_request_pulse_input_action_result, serialize_action_result

    case_state = {"chief_complaint": "腹痛"}
    plan = {"missing_slots": []}
    action_result = build_request_pulse_input_action_result(case_state=case_state, plan=plan)
    payload = serialize_action_result(action_result)

    checks = [
        ("langgraph pulse builder action name", payload.get("name") == "request_pulse_input"),
        ("langgraph pulse builder status", payload.get("status") == "WAITING_PULSE_INPUT"),
        ("langgraph pulse builder render mode", payload.get("render_mode") == "followup"),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试跟进策略的构建函数是否能够正确生成针对单个信息项和多个信息项的跟进行动结果
def test_followup_action_builders():
    from agent.runtime_utils import (
        build_followup_bundle_action_result,
        build_followup_single_action_result,
        serialize_action_result,
    )

    single_case_state = {"followup_counts": {}, "chief_complaint": "胸痛"}
    single_plan = {"missing_slots": ["severity"]}
    single_payload = serialize_action_result(
        build_followup_single_action_result(single_case_state, single_plan)
    )

    bundle_case_state = {"chief_complaint": "腹痛"}
    bundle_plan = {
        "missing_slots": ["cold_heat", "appetite", "sleep"],
        "four_diagnosis_focus": {"问": ["寒热倾向", "食欲与纳食", "睡眠情况"]},
    }
    bundle_payload = serialize_action_result(
        build_followup_bundle_action_result(bundle_case_state, bundle_plan)
    )

    checks = [
        ("followup single builder action name", single_payload.get("name") == "ask_followup_single"),
        ("followup single builder status", single_payload.get("status") == "COLLECTING_INFO"),
        ("followup single builder has slot", single_payload.get("followup_slot") == "severity"),
        ("followup bundle builder action name", bundle_payload.get("name") == "ask_followup_bundle"),
        ("followup bundle builder status", bundle_payload.get("status") == "COLLECTING_INFO"),
        ("followup bundle builder keeps missing slots", bundle_payload.get("missing_slots") == ["cold_heat", "appetite", "sleep"]),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试当使用langgraph运行时时
# 系统在图状态中正确记录和处理冲突澄清的action subgraph输出
def test_langgraph_clarify_conflict_route():
    from agent.factory import create_agent, is_langgraph_available
    from llm.llm import LLM
    from llm.prompt import SYSTEM_PROMPT
    from memory.memory import ConversationMemory

    if not is_langgraph_available():
        print("[SKIP] langgraph clarify route (langgraph not installed)")
        return True

    memory = ConversationMemory()
    agent = create_agent(
        llm=LLM(system_prompt=SYSTEM_PROMPT),
        memory=memory,
        runtime="langgraph",
    )
    agent.run("我今年32岁")
    agent.run("其实我28岁")
    graph_state = agent.get_graph_state()
    values = getattr(graph_state, "values", {})
    action_payload = values.get("action_result", {})

    checks = [
        ("langgraph conflict route serialized as dict", isinstance(action_payload, dict)),
        ("langgraph conflict route resolved", action_payload.get("name") == "clarify_conflict"),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试当使用langgraph运行时时
# 系统在图状态中正确记录和处理跟进提问的action subgraph输出
def test_langgraph_followup_route():
    agent = _create_langgraph_agent()
    if agent is None:
        return True

    response = agent.run("我发热，还咳嗽")
    snapshot = agent.get_case_snapshot()
    graph_state = agent.get_graph_state()
    values = getattr(graph_state, "values", {})
    action_payload = values.get("action_result", {})

    checks = [
        ("langgraph followup action serialized as dict", isinstance(action_payload, dict)),
        ("langgraph followup route resolved", action_payload.get("name") in {"ask_followup_single", "ask_followup_bundle"}),
        ("langgraph followup status recorded", snapshot.get("status") == "COLLECTING_INFO"),
        ("langgraph followup response generated", bool(response.strip())),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试工具协议：ToolResult 结构完整，装饰器捕获异常后降级为 status=error
def test_tool_protocol_result_and_error_fallback():
    from tools.protocol import build_tool_result, managed_tool

    result = build_tool_result("demo_tool", "1.0", {"a": 1}, elapsed_ms=3.14159)

    @managed_tool("broken_tool", "1.0", "用于测试异常降级的工具")
    def _broken_tool(payload):
        raise ValueError("boom")

    error_result = _broken_tool({})

    checks = [
        ("tool result status ok", result["status"] == "ok"),
        ("tool result carries data", result["data"] == {"a": 1}),
        ("tool result carries version", result["version"] == "1.0"),
        ("tool result elapsed rounded", result["elapsed_ms"] == 3.14),
        ("broken tool status error", error_result["status"] == "error"),
        ("broken tool error message", "boom" in error_result["error"]),
        ("broken tool name kept", error_result["tool"] == "broken_tool"),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试工具注册表：三个核心工具均已注册，清单可序列化，兼容层行为保持不变
def test_tool_registry_and_compat_layer():
    from tools.registry import default_registry
    from tools.risk_tool import risk_assessment

    names = {item["name"] for item in default_registry.list_tools()}
    risk_result = risk_assessment({"red_flags": ["持续胸痛"], "symptoms": []})

    checks = [
        ("registry lists risk_assessment", "risk_assessment" in names),
        ("registry lists guideline", "guideline" in names),
        ("registry lists symptom_extraction", "symptom_extraction" in names),
        ("registry get returns callable", callable(default_registry.get("risk_assessment"))),
        ("compat risk_assessment returns plain dict", isinstance(risk_result, dict) and "status" not in risk_result),
        ("compat risk_assessment keeps HIGH behavior", risk_result.get("risk") == "HIGH"),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试事件流：EventLogger 写入/读取 JSONL，支持 session 过滤；metrics 汇总正确
def test_event_logger_and_metrics_summary():
    import tempfile

    from observability.events import EventLogger
    from observability.logger import set_trace_context
    from observability.metrics import summarize_events

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = EventLogger(tmp_dir)
        set_trace_context(session_id="sess-test", turn_id="turn-1")
        try:
            logger.emit("llm_call", provider="mock", elapsed_ms=12.0, fallback=True)
            logger.emit("node_exit", node="plan", elapsed_ms=2.5)
            logger.emit("risk_assessed", risk="LOW", matched_rules=[])
        finally:
            set_trace_context(session_id="", turn_id="")

        events = logger.read_events(session_id="sess-test")
        all_events = logger.read_events()
        summary = summarize_events(events)

        checks = [
            ("events written and read back", len(events) == 3),
            ("events carry session_id", events[0]["session_id"] == "sess-test"),
            ("events carry turn_id", events[0]["turn_id"] == "turn-1"),
            ("session filter works", len(all_events) == 3),
            ("summary counts llm calls", summary["llm_calls"] == 1),
            ("summary counts mock fallback", summary["mock_fallbacks"] == 1),
            ("summary aggregates node stats", summary["node_stats"].get("plan", {}).get("count") == 1),
            ("summary counts risk distribution", summary["risk_distribution"].get("LOW") == 1),
        ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试 LLM 埋点：get_runtime_status 返回 metrics，mock 调用被正确计数
def test_llm_metrics():
    from llm.llm import LLM
    from llm.prompt import SYSTEM_PROMPT

    llm = LLM(system_prompt=SYSTEM_PROMPT, provider="mock")
    llm.call("演示调用")
    status = llm.get_runtime_status()
    metrics = status.get("metrics", {})

    checks = [
        ("runtime status contains metrics", "metrics" in status),
        ("metrics counts call", metrics.get("call_count") == 1),
        ("metrics counts mock fallback", metrics.get("mock_fallback_count") == 1),
        ("metrics fallback rate", metrics.get("mock_fallback_rate") == 1.0),
        ("metrics latency recorded", metrics.get("total_latency_ms", 0) >= 0),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


def main():
    results = [
        test_default_runtime_prefers_langgraph(),
        test_imports(),
        test_agent_flow(),
        test_high_risk_escalation(),
        test_followup_strategy(),
        test_tcm_evidence_and_pulse(),
        test_optional_slot_not_repeated(),
        test_pulse_prompt_does_not_repeat(),
        test_long_term_profile_promotion(),
        test_profile_store_reuses_long_term_profile_across_sessions(),
        test_session_store_reuses_session_state_across_memory_instances(),
        test_conversation_summary_and_open_questions(),
        test_langgraph_runtime_smoke(),
        test_langgraph_high_risk_escalation(),
        test_langgraph_pulse_prompt_does_not_repeat(),
        test_langgraph_checkpointer_thread_state(),
        test_langgraph_action_subgraph_routes(),
        test_langgraph_pulse_action_subgraph_route(),
        test_followup_action_builders(),
        test_langgraph_clarify_conflict_route(),
        test_langgraph_followup_route(),
        test_tool_protocol_result_and_error_fallback(),
        test_tool_registry_and_compat_layer(),
        test_event_logger_and_metrics_summary(),
        test_llm_metrics(),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\nSummary: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
