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
        "tools.knowledge_tool",
        "evaluation.run_eval",
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
        ("response asks for more detail", any(keyword in response for keyword in ("补充", "持续", "严重程度", "部位", "多久", "多久了", "多重", "程度", "严重", "怎么个不舒服法", "怎么个不舒服"))),
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

    planner = Planner()
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

# 测试手填基本信息与长期记忆绑定：apply_manual_profile 写入画像并同步当前会话，
# 不写 slot_history 不触发矛盾检测；新会话从画像预填空字段，旧会话已有值不覆盖；
# delete_profile 清除后回落到默认画像
def test_manual_profile_save_and_prefill():
    from memory.memory import ConversationMemory
    from memory.profile_store import InMemoryProfileStore
    from memory.session_store import InMemorySessionStore

    profile_store = InMemoryProfileStore()
    memory = ConversationMemory(profile_store=profile_store, user_id="manual-user")
    memory.apply_manual_profile(
        {
            "age": "45",
            "sex": "男",
            "past_history": ["高血压", "糖尿病"],
            "allergy_history": ["青霉素"],
            "medication_history": ["阿司匹林"],
        }
    )
    profile = memory.get_long_term_profile()
    case_state = memory.get_case_state()

    # 同用户新会话：长期画像预填空字段
    second_case_state = ConversationMemory(
        profile_store=profile_store, user_id="manual-user"
    ).get_case_state()

    # 旧会话恢复：已有值不被画像覆盖（直接写 case_state 避免 update_case 的画像提升干扰）
    session_store = InMemorySessionStore()
    preset_memory = ConversationMemory(
        profile_store=profile_store,
        user_id="manual-user",
        session_store=session_store,
        session_id="restored-session",
    )
    preset_memory.case_state["age"] = "60"
    preset_memory._persist_session_state()
    restored_state = ConversationMemory(
        profile_store=profile_store,
        user_id="manual-user",
        session_store=session_store,
        session_id="restored-session",
    ).get_case_state()

    # 清除长期记忆后回落默认画像
    profile_store.delete_profile("manual-user")
    deleted = profile_store.get_profile("manual-user", memory._default_profile())

    checks = [
        ("manual profile saves scalars", profile["age"] == "45" and profile["sex"] == "男"),
        ("manual profile saves lists", "高血压" in profile["past_history"] and "青霉素" in profile["allergy_history"]),
        ("manual profile summary generated", profile["profile_summary"] != "长期画像暂未形成"),
        ("manual profile syncs case state", case_state["age"] == "45" and "糖尿病" in case_state["past_history"]),
        ("manual profile skips slot history", "age" not in case_state["slot_history"] and not case_state["contradictions"]),
        ("new session prefills from profile", second_case_state["age"] == "45" and "高血压" in second_case_state["past_history"]),
        ("restored session values not overwritten", restored_state["age"] == "60"),
        ("delete profile returns default", deleted["age"] == "" and deleted["profile_summary"] == "长期画像暂未形成"),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试手填画像 API 往返：POST 规范化分隔字符串并写入画像与会话，GET 读回，clear 仅清长期文件
def test_profile_api_roundtrip():
    import app as app_module
    from memory.profile_store import InMemoryProfileStore
    from memory.session_store import InMemorySessionStore

    original_profile_store = app_module.profile_store
    original_session_store = app_module.session_store
    # 换成内存存储，避免测试污染 data/profiles 与 data/sessions
    app_module.profile_store = InMemoryProfileStore()
    app_module.session_store = InMemorySessionStore()
    try:
        client = app_module.app.test_client()
        post = client.post(
            "/api/profile",
            json={
                "age": "70",
                "sex": "女",
                "past_history": "高血压, 糖尿病",
                "allergy_history": ["青霉素"],
                "medication_history": "",
            },
        )
        post_data = post.get_json() or {}
        get_data = client.get("/api/profile").get_json() or {}
        clear_data = client.post("/api/profile/clear").get_json() or {}
    finally:
        app_module.profile_store = original_profile_store
        app_module.session_store = original_session_store
        app_module.session_cache.clear()

    saved = post_data.get("profile", {})
    checks = [
        ("profile api save ok", post.status_code == 200 and post_data.get("ok")),
        ("profile api splits delimited string", saved.get("past_history") == ["高血压", "糖尿病"]),
        ("profile api keeps array field", saved.get("allergy_history") == ["青霉素"]),
        ("profile api blank list becomes empty", saved.get("medication_history") == []),
        ("profile api read back", get_data.get("age") == "70" and get_data.get("sex") == "女"),
        ("profile api clear resets", clear_data.get("ok") and clear_data.get("profile", {}).get("age") == ""),
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


# 测试知识层外置数据：JSON 加载校验通过，口语别名能归一为规范术语
def test_knowledge_data_loading():
    from knowledge.tcm_knowledge import (
        CHIEF_COMPLAINT_PRIORITIES,
        KNOWLEDGE_VERSION,
        SYNDROME_RULES,
        get_red_flags,
        get_syndrome_advice,
        normalize_term,
    )

    advice = get_syndrome_advice("风寒束表")
    checks = [
        ("syndrome rules expanded", len(SYNDROME_RULES) >= 15),
        ("syndrome rule carries advice", bool(advice["treatment_principle"]) and bool(advice["lifestyle_advice"])),
        ("chief complaint priorities expanded", len(CHIEF_COMPLAINT_PRIORITIES) >= 10),
        ("red flags loaded from knowledge data", len(get_red_flags()) >= 5),
        ("knowledge versions tracked", "syndrome_rules" in KNOWLEDGE_VERSION and "red_flags" in KNOWLEDGE_VERSION),
        ("colloquial terms normalized", normalize_term("怕冷，没胃口") == "恶寒，纳差"),
        ("unknown text untouched", normalize_term("随意描述") == "随意描述"),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试知识检索工具：协议版返回标准 ToolResult，兼容层返回数据 dict，工具已注册
def test_knowledge_retrieval_tool():
    from tools.knowledge_tool import search_knowledge, search_knowledge_tool
    from tools.registry import default_registry

    result = search_knowledge_tool("风寒束表")
    compat = search_knowledge("咳嗽怕冷")
    names = {item["name"] for item in default_registry.list_tools()}

    checks = [
        ("protocol result ok", result["status"] == "ok" and result["tool"] == "knowledge_retrieval"),
        ("hits syndrome by exact name", any(hit["type"] == "syndrome" and hit["name"] == "风寒束表" for hit in result["data"]["hits"])),
        ("compat layer returns data dict", isinstance(compat, dict) and compat["total"] > 0),
        ("knowledge tool registered", "knowledge_retrieval" in names),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试 Planner 规则增强：阈值配置化、主诉模糊匹配、证型定向追问加权
def test_planner_rule_enhancements():
    from agent.planner import PLANNER_CONFIG, Planner

    planner = Planner()
    fuzzy = planner._chief_complaint_priorities("咳嗽得厉害")
    boosted = planner._syndrome_focus_slots([{"name": "风寒束表", "score": 6}])
    low_score = planner._syndrome_focus_slots([{"name": "风寒束表", "score": 2}])

    checks = [
        ("planner config exposes thresholds", PLANNER_CONFIG.get("confidence_ready_threshold") == 0.75),
        ("chief complaint fuzzy match", "cold_heat" in fuzzy),
        ("syndrome focus boosts evidence slots", bool(boosted)),
        ("low score syndrome not boosted", not low_score),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试 Planner review 拦截规则：高风险不宜继续追问/等脉诊，完整度不足不宜过早请求脉诊
def test_planner_review_interceptions():
    from agent.planner import Planner
    from agent.router import ActionResult

    planner = Planner()
    followup_action = ActionResult(name="ask_followup", response="", status="", is_final=False, missing_slots=["sleep"])
    review_high = planner.review_action({}, {"missing_slots": ["sleep"], "confidence": 0.5}, {"risk": "HIGH"}, followup_action)

    pulse_action = ActionResult(name="request_pulse_input", response="", status="", is_final=False, missing_slots=[])
    review_pulse = planner.review_action({}, {"completion_score": 0.3}, {"risk": "LOW"}, pulse_action)

    checks = [
        ("high risk followup intercepted", review_high["needs_replan"] and review_high["suggested_action"] == "risk_escalation"),
        ("early pulse request intercepted", review_pulse["needs_replan"] and review_pulse["suggested_action"] == "ask_followup_bundle"),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 评测集 smoke：回放一个带标注用例走真实图链路，防止评测基线回归
def test_evaluation_case_smoke():
    from evaluation.run_eval import evaluate_case, load_cases
    from evaluation.run_eval import main as eval_main

    cases = load_cases(case_filter="01_high_risk_chest_pain")
    if not cases:
        print("[FAIL] evaluation case 01_high_risk_chest_pain not found")
        return False

    result = evaluate_case(cases[0])
    # B4：--provider 参数解析并透传（默认 mock 行为不变）
    provider_arg_ok = eval_main(["--case", "01_high_risk_chest_pain", "--provider", "mock"]) == 0
    checks = [
        ("case loaded", result["id"] == "01_high_risk_chest_pain"),
        ("case passes all assertions", result["pass"]),
        ("eval provider argument parsed and runs", provider_arg_ok),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}: {result.get('failures')}")
    return passed


# 测试矛盾消除机制：标量字段前后不一致时产生矛盾，重新确认后 resolve 应消除矛盾
def test_contradiction_resolution():
    from memory.memory import ConversationMemory

    memory = ConversationMemory()
    memory.update_case({"age": "25", "symptoms": ["腹痛"]})
    memory.update_case({"age": "65"})
    has_contradiction = bool(memory.get_case_state()["contradictions"])

    memory.resolve_contradiction("age")
    state = memory.get_case_state()
    checks = [
        ("contradiction detected on conflicting age", has_contradiction),
        ("contradiction resolved after reconfirmation", not state["contradictions"]),
        ("latest value kept after resolve", state["slot_history"]["age"] == ["65"]),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed

# 测试空槽位防御：缺失队列为空（填满或达限暂缓）时直接给最终建议，不再生成空追问
def test_planner_no_empty_bundle():
    from agent.planner import Planner

    planner = Planner()
    case_state = {
        "chief_complaint": "腹痛",
        "symptoms": ["腹痛"],
        "duration": "两天",
        "severity": "中度",
        "location": "腹部",
        "cold_heat": "怕冷",
        "appetite": "纳差",
        "sleep": "失眠",
        "stool_urine": "便秘",
        "pain_character": "隐痛",
        "pulse_declined": True,
        "followup_counts": {"sweating": 1, "thirst": 1, "emotion": 1},
        "risk_level": "LOW",
        "summary": "症状: 腹痛",
        "tcm_summary": "寒热: 怕冷",
        "accompanying_symptoms": [],
    }
    plan = planner.create_plan(case_state)

    checks = [
        ("no missing slots when filled or deferred", not plan["missing_slots"]),
        ("final advice instead of empty bundle", plan["next_action"] == "final_advice"),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}, actual next_action={plan['next_action']}")
    return passed

# 测试 RAG 前置语料与检索：语料 schema 校验、BM25 命中排序、knowledge_tool 合并输出
def test_corpus_knowledge_and_retrieval():
    from knowledge.tcm_knowledge import KNOWLEDGE_VERSION, SYNDROME_RULES, get_corpus
    from knowledge.retriever import DEFAULT_RETRIEVER
    from tools.knowledge_tool import search_knowledge

    corpus = get_corpus()
    ids = [entry["id"] for entry in corpus]
    syndrome_names = {rule["name"] for rule in SYNDROME_RULES}
    schema_ok = (
        len(corpus) >= 30
        and len(set(ids)) == len(ids)
        and all(entry.get("title") and entry.get("content") and entry.get("tags") for entry in corpus)
        and all(s in syndrome_names for entry in corpus for s in entry.get("syndromes", []))
        and {"formulas", "health_advice", "faq"} <= set(KNOWLEDGE_VERSION)
    )

    formula_top = DEFAULT_RETRIEVER.search("桂枝汤", top_k=1)
    faq_top = DEFAULT_RETRIEVER.search("感冒了能喝姜汤吗", top_k=1)
    advice_hits = DEFAULT_RETRIEVER.search("风寒束表调护", top_k=5)
    empty_hits = DEFAULT_RETRIEVER.search("qwerty", top_k=3)

    merged = search_knowledge("柴胡疏肝散")
    merged_corpus = [hit for hit in merged["hits"] if hit["type"] == "corpus"]

    # RAG 注入：content 透传到检索命中，最终建议按证型+主诉附语料参考
    from agent.runtime_utils import build_final_advice_action_result

    content_passthrough_ok = (
        bool(formula_top) and bool(formula_top[0].get("content"))
        and bool(merged_corpus) and all(hit.get("content") for hit in merged_corpus)
    )

    enriched_advice = build_final_advice_action_result(
        {"chief_complaint": "咳嗽", "symptoms": ["咳嗽"], "tcm_summary": "恶寒无汗"},
        {"risk": "LOW"},
        {"summary": "测试建议", "advice": []},
        {"syndrome_candidates": [{"name": "风寒束表"}], "completion_label": "基本完成", "confidence": 0.8},
    )
    expected_contents = [
        hit["content"]
        for hit in search_knowledge("风寒束表 咳嗽", top_k=6)["hits"]
        if hit["type"] == "corpus" and hit.get("content")
    ][:3]
    injection_ok = (
        bool(expected_contents)
        and "知识库参考" in enriched_advice.response
        and "知识库版本" in enriched_advice.response
        and all(content in enriched_advice.response for content in expected_contents)
    )

    bare_advice = build_final_advice_action_result(
        {"symptoms": ["头晕"], "tcm_summary": ""},
        {"risk": "LOW"},
        {"summary": "测试建议", "advice": []},
        {"syndrome_candidates": [], "completion_label": "待评估", "confidence": 0.2},
    )
    no_injection_ok = "知识库参考" not in bare_advice.response

    # R2：LLM prompt 知识上下文注入——真实 provider 路径 prompt 附检索参考，mock 路径原样返回草稿不调 LLM
    from agent.router import ActionResult
    from agent.runtime_utils import render_response

    class _RecordingLLM:
        def __init__(self, provider):
            self.last_provider_used = provider
            self.prompts = []

        def call(self, prompt):
            self.prompts.append(prompt)
            return "LLM 渲染回复"

    class _SimpleMemory:
        def get_prompt_context_text(self):
            return "对话上下文"

        def get_profile_context_text(self):
            return "画像上下文"

    final_action = ActionResult(
        name="final_advice",
        response=enriched_advice.response,
        status="GENERATING_ADVICE",
        is_final=True,
        missing_slots=[],
        render_mode="final",
    )
    case_with_syndrome = {"chief_complaint": "咳嗽", "symptoms": ["咳嗽"]}
    plan_with_syndrome = {"syndrome_candidates": [{"name": "风寒束表"}]}

    deepseek_llm = _RecordingLLM("deepseek")
    rendered = render_response(
        deepseek_llm, _SimpleMemory(), final_action,
        case_with_syndrome, {"risk": "LOW"}, {"summary": "测试建议"}, plan_with_syndrome,
    )
    expected_prompt_hits = [
        hit for hit in search_knowledge("风寒束表 咳嗽", top_k=6)["hits"]
        if hit["type"] == "corpus" and hit.get("content")
    ][:3]
    prompt_context_ok = (
        rendered == "LLM 渲染回复"
        and len(deepseek_llm.prompts) == 1
        and "knowledge_context:" in deepseek_llm.prompts[0]
        and all(hit["name"] in deepseek_llm.prompts[0] and hit["content"] in deepseek_llm.prompts[0] for hit in expected_prompt_hits)
    )

    mock_llm = _RecordingLLM("mock")
    mock_rendered = render_response(
        mock_llm, _SimpleMemory(), final_action,
        case_with_syndrome, {"risk": "LOW"}, {"summary": "测试建议"}, plan_with_syndrome,
    )
    mock_path_ok = mock_rendered == final_action.response and not mock_llm.prompts

    bare_llm = _RecordingLLM("deepseek")
    render_response(
        bare_llm, _SimpleMemory(), final_action,
        {"symptoms": ["头晕"]}, {"risk": "LOW"}, {"summary": "测试建议"}, {"syndrome_candidates": []},
    )
    empty_context_ok = bool(bare_llm.prompts) and "knowledge_context:\n无" in bare_llm.prompts[0]

    # A1：脉诊请求直出固定文案，不走 LLM 重写（实测重写会改写成雷同追问，语义丢失）
    pulse_action = ActionResult(
        name="request_pulse_input",
        response="脉诊请求固定文案",
        status="WAITING_PULSE_INPUT",
        is_final=False,
        missing_slots=[],
        render_mode="followup",
    )
    pulse_llm = _RecordingLLM("deepseek")
    pulse_rendered = render_response(
        pulse_llm, _SimpleMemory(), pulse_action,
        case_with_syndrome, {"risk": "LOW"}, {"summary": "测试建议"}, plan_with_syndrome,
    )
    pulse_direct_ok = pulse_rendered == "脉诊请求固定文案" and not pulse_llm.prompts

    # A2：最终回复 prompt 要求原样保留知识库参考段，避免 LLM 重写抹掉出处与版本信息
    citation_kept_clause_ok = (
        bool(deepseek_llm.prompts)
        and "知识库参考" in deepseek_llm.prompts[0]
        and "原样保留" in deepseek_llm.prompts[0]
    )

    # C5/C8：语料可被加载且确定性命中（方剂 44/调护 32/FAQ 14 = 90 条，含 MIT 开源引入 32 条）
    from knowledge.tcm_knowledge import get_corpus

    corpus_entries = get_corpus()
    new_formula_top = search_knowledge("小柴胡汤", top_k=1)["hits"]
    new_faq_top = search_knowledge("中药应该怎么煎服", top_k=1)["hits"]
    corpus_expansion_ok = (
        len(corpus_entries) == 90
        and bool(new_formula_top) and new_formula_top[0]["id"] == "f_xiao_chaihu_tang"
        and bool(new_faq_top) and new_faq_top[0]["id"] == "faq_how_to_decoct"
    )

    # C8：MIT 引入经方确定性 top-1 命中，且来源标注字段在场（许可合规要求）
    mit_formula_top = search_knowledge("真武汤", top_k=1)["hits"]
    mit_entry = next((e for e in corpus_entries if e["id"] == "f_zhenwu_tang"), None)
    mit_import_ok = (
        bool(mit_formula_top) and mit_formula_top[0]["id"] == "f_zhenwu_tang"
        and mit_entry is not None
        and "MIT License" in mit_entry.get("source", "")
    )

    # R3：TF-IDF 对比后端——与 BM25 同金标准的确定性排名，命中结构一致（含 content）
    from knowledge.retriever import TFIDFRetriever

    tfidf = TFIDFRetriever()
    tfidf_formula_top = tfidf.search("桂枝汤", top_k=1)
    tfidf_faq_top = tfidf.search("感冒了能喝姜汤吗", top_k=1)
    tfidf_ok = (
        bool(tfidf_formula_top) and tfidf_formula_top[0]["id"] == "f_guizhi_tang"
        and bool(tfidf_faq_top) and tfidf_faq_top[0]["id"] == "faq_cold_ginger_soup"
        and tfidf.search("qwerty", top_k=3) == []
        and tfidf.search("", top_k=3) == []
        and all(
            {"id", "title", "category", "score", "source", "content"} <= set(hit)
            for hit in tfidf_formula_top + tfidf_faq_top
        )
    )

    # 对比脚本统计函数单测：MRR 与平均排名计算正确，空列表降级为 0
    from evaluation.compare_retrievers import summarize

    mrr, avg_rank = summarize([1, 2, 4])
    stats_ok = (
        abs(mrr - (1 + 0.5 + 0.25) / 3) < 1e-9
        and abs(avg_rank - 7 / 3) < 1e-9
        and summarize([]) == (0.0, 0.0)
    )

    checks = [
        ("corpus schema validated with syndrome linkage", schema_ok),
        ("formula query ranks exact formula first", bool(formula_top) and formula_top[0]["id"] == "f_guizhi_tang"),
        ("faq query ranks matching answer first", bool(faq_top) and faq_top[0]["id"] == "faq_cold_ginger_soup"),
        (
            "advice query hits linked advice entries",
            any(hit["category"] in {"饮食", "起居"} and hit["id"].startswith("ha_wind_cold") for hit in advice_hits),
        ),
        ("noisy query returns empty", empty_hits == []),
        (
            "knowledge tool merges corpus hits",
            any(hit["name"] == "柴胡疏肝散" for hit in merged_corpus) and all(hit.get("source") for hit in merged_corpus),
        ),
        ("retrieval hits carry corpus content for injection", content_passthrough_ok),
        ("final advice injects corpus reference for matched syndrome", injection_ok),
        ("final advice skips injection without syndrome candidates", no_injection_ok),
        ("real provider prompt carries knowledge context", prompt_context_ok),
        ("mock provider returns draft without llm call", mock_path_ok),
        ("empty retrieval passes placeholder context", empty_context_ok),
        ("pulse request skips llm rewrite", pulse_direct_ok),
        ("final prompt keeps knowledge citation clause", citation_kept_clause_ok),
        ("corpus expansion loads and retrieves deterministically", corpus_expansion_ok),
        ("mit-sourced formulas carry attribution and rank top-1", mit_import_ok),
        ("tfidf backend ranks gold queries deterministically", tfidf_ok),
        ("compare script rank statistics correct", stats_ok),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试 MCP 接入基础：配置加载校验、非法配置报错、未连接时调用降级为 error 而非异常
def test_mcp_config_and_adapter_degradation():
    import json
    import sys
    import tempfile
    from pathlib import Path

    from mcp_bridge.adapter import build_mcp_tool, register_server_tools, validate_arguments
    from mcp_bridge.client import MCPClientError, MCPClientManager, build_server_env
    from mcp_bridge.config import MCPConfigError, load_mcp_config
    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as tmp:
        valid_path = Path(tmp) / "valid.json"
        valid_path.write_text(
            json.dumps(
                {
                    "version": "1.0.0",
                    "servers": [
                        {"name": "demo", "transport": "stdio", "command": "python", "args": [], "env": {}, "enabled": True, "call_timeout": 5}
                    ],
                }
            ),
            encoding="utf-8",
        )
        valid = load_mcp_config(valid_path)
        command_normalized = valid[0]["command"] == sys.executable
        timeout_parsed = valid[0].get("call_timeout") == 5.0

        bad_transport = Path(tmp) / "bad_transport.json"
        bad_transport.write_text(
            json.dumps({"version": "1.0.0", "servers": [{"name": "x", "transport": "ftp", "enabled": True, "command": "python"}]}),
            encoding="utf-8",
        )
        bad_enabled = Path(tmp) / "bad_enabled.json"
        bad_enabled.write_text(
            json.dumps({"version": "1.0.0", "servers": [{"name": "x", "transport": "stdio", "enabled": "yes", "command": "python"}]}),
            encoding="utf-8",
        )
        reserved_transport = Path(tmp) / "reserved_transport.json"
        reserved_transport.write_text(
            json.dumps({"version": "1.0.0", "servers": [{"name": "x", "transport": "streamable_http", "enabled": True}]}),
            encoding="utf-8",
        )
        bad_timeout = Path(tmp) / "bad_timeout.json"
        bad_timeout.write_text(
            json.dumps({"version": "1.0.0", "servers": [{"name": "x", "transport": "stdio", "enabled": True, "command": "python", "call_timeout": -1}]}),
            encoding="utf-8",
        )
        transport_error = enabled_error = reserved_message = timeout_error = False
        try:
            load_mcp_config(bad_transport)
        except MCPConfigError:
            transport_error = True
        try:
            load_mcp_config(bad_enabled)
        except MCPConfigError:
            enabled_error = True
        try:
            load_mcp_config(reserved_transport)
        except MCPConfigError as exc:
            reserved_message = "预留" in str(exc)
        try:
            load_mcp_config(bad_timeout)
        except MCPConfigError:
            timeout_error = True

    # 项目自带配置可加载：hospital_locator 启用（含服务级超时）、pulse_device 占位禁用
    project_servers = {entry["name"]: entry for entry in load_mcp_config()}
    project_config_ok = (
        project_servers.get("hospital_locator", {}).get("enabled") is True
        and project_servers.get("hospital_locator", {}).get("call_timeout") == 5.0
        and project_servers.get("pulse_device", {}).get("enabled") is False
    )

    # 子进程环境最小化：不透传宿主全量环境，强制 UTF-8，配置 env 可覆盖默认
    import os

    os.environ["MEDICAL_FAKE_SECRET"] = "should-not-leak"
    os.environ["AMAP_API_KEY"] = "fake-amap-key"
    os.environ["HOSPITAL_DATA_URL"] = "http://127.0.0.1:9/hospitals.json"
    try:
        minimal = build_server_env({})
        env_minimized = (
            "MEDICAL_FAKE_SECRET" not in minimal
            and "PATH" in minimal
            and minimal.get("PYTHONUTF8") == "1"
            and minimal.get("PYTHONIOENCODING") == "utf-8"
        )
        # C7：hospital_locator 数据源键点名透传给子进程（否则实际 stdio 部署收不到配置）
        env_datasource_passthrough_ok = (
            minimal.get("AMAP_API_KEY") == "fake-amap-key"
            and minimal.get("HOSPITAL_DATA_URL") == "http://127.0.0.1:9/hospitals.json"
        )
        overridden = build_server_env({"PYTHONIOENCODING": "gbk", "CUSTOM_VAR": "v"})
        env_override_ok = overridden["PYTHONIOENCODING"] == "gbk" and overridden["CUSTOM_VAR"] == "v"
    finally:
        os.environ.pop("MEDICAL_FAKE_SECRET", None)
        os.environ.pop("AMAP_API_KEY", None)
        os.environ.pop("HOSPITAL_DATA_URL", None)

    # 未启动的管理器：适配工具返回标准 error ToolResult 而非抛异常
    manager = MCPClientManager()
    tool = build_mcp_tool(manager, "demo_server", {"name": "demo_tool", "description": "演示"})
    result = tool(foo="bar")
    degraded_ok = result["status"] == "error" and result["error"]
    manager_error_ok = False
    try:
        manager.call_tool("demo_server", "demo_tool", {})
    except MCPClientError:
        manager_error_ok = True

    # 命名冲突防护：同名工具跳过不覆盖，不冲突的正常注册
    fresh_registry = ToolRegistry()

    def _original():
        return "original"

    _original.tool_name = "demo_server_demo_tool"
    _original.tool_version = "0"
    _original.tool_description = "原有工具"
    fresh_registry.register(_original)
    registered_count = register_server_tools(
        object(), fresh_registry, "demo_server",
        [{"name": "demo_tool"}, {"name": "other_tool"}],
    )
    collision_guard_ok = (
        registered_count == 1
        and fresh_registry.get("demo_server_demo_tool")() == "original"
        and fresh_registry.get("demo_server_other_tool") is not None
    )

    # input_schema 本地校验：非法参数在本地拦截，不发起远端调用
    schema = {
        "type": "object",
        "properties": {"location": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["location"],
    }
    schema_checks = (
        validate_arguments(schema, {"location": "北京"}) == ""
        and "必填" in validate_arguments(schema, {})
        and "未知参数" in validate_arguments(schema, {"location": "北京", "foo": 1})
        and "类型" in validate_arguments(schema, {"location": 123})
        and "类型" in validate_arguments(schema, {"location": "北京", "limit": True})
        and validate_arguments({}, {"anything": 1}) == ""
    )

    class _RecordingManager:
        def __init__(self):
            self.calls = []

        def call_tool(self, server, tool, arguments=None, timeout=None):
            self.calls.append((server, tool, arguments))
            return {"ok": True}

    recording = _RecordingManager()
    guarded_tool = build_mcp_tool(
        recording, "demo_server",
        {"name": "guarded", "description": "校验", "input_schema": schema},
    )
    blocked = guarded_tool(limit=1)
    allowed = guarded_tool(location="北京")
    schema_guard_ok = (
        schema_checks
        and blocked["status"] == "error" and "必填" in blocked["error"]
        and allowed["status"] == "ok"
        and recording.calls == [("demo_server", "guarded", {"location": "北京"})]
    )

    checks = [
        ("valid mcp config parsed with interpreter normalization", command_normalized),
        ("call_timeout parsed as positive float", timeout_parsed),
        ("invalid transport rejected", transport_error),
        ("non-bool enabled rejected", enabled_error),
        ("reserved streamable_http rejected with dedicated message", reserved_message),
        ("non-positive call_timeout rejected", timeout_error),
        ("project mcp config loads with expected flags", project_config_ok),
        ("server env minimized with utf-8 defaults", env_minimized),
        ("data source env keys passed through to server env", env_datasource_passthrough_ok),
        ("config env overrides defaults", env_override_ok),
        ("adapter degrades to error result when disconnected", bool(degraded_ok)),
        ("manager raises MCPClientError when not started", manager_error_ok),
        ("name collision skipped without overwriting", collision_guard_ok),
        ("schema validation blocks bad args locally", schema_guard_ok),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试 MCP 端到端：真实启动仓库内 mock 地图服务，调用返回结构正确，HIGH 升级回复可附加医院清单
def test_mcp_end_to_end_hospital_locator():
    import json
    import os

    from agent.runtime_utils import build_risk_escalation_action_result
    from mcp_bridge.adapter import connect_mcp_servers, shutdown_mcp_servers
    from mcp_bridge.client import default_manager
    from tools.registry import default_registry

    connect_mcp_servers()
    os.environ.pop("MEDICAL_AGENT_LOCATION", None)
    try:
        connected = "hospital_locator" in default_manager.connected_servers()
        # 连接后状态暴露：connected=True 且计数器存在
        server_status = default_manager.status()["servers"].get("hospital_locator", {})
        status_ok = server_status.get("connected") is True and "calls" in server_status

        tool = default_registry.get("hospital_locator_search_nearby_hospitals")
        result = tool(location="北京", department="急诊") if tool else {"status": "error"}
        # 数据源无关断言：配置高德 key 时走真实 POI（无 distance_km/科室信息），未配置走演示数据
        result_note = (result.get("data") or {}).get("note", "")
        is_amap = "高德" in result_note
        result_hospitals = (result.get("data") or {}).get("hospitals", [])
        data_ok = (
            result.get("status") == "ok"
            and result["data"].get("count", 0) >= 2
            and all(h.get("name") for h in result_hospitals)
            and (is_amap or all("distance_km" in h for h in result_hospitals))
        )
        unknown = tool(location="不存在的城市") if tool else {"data": {}}
        unknown_ok = unknown.get("status") == "ok" and unknown["data"].get("count", 0) == 0

        # 成功调用后指标计数：calls 至少 2 次（前两次查询），errors 为 0
        after_calls = default_manager.status()["servers"].get("hospital_locator", {})
        metrics_ok = after_calls.get("calls", 0) >= 2 and after_calls.get("errors", 0) == 0

        # 确定性重连验证：把 session 置为失效对象模拟子进程崩溃，调用应自动重连并成功
        default_manager._servers["hospital_locator"]["session"] = None
        recovered = tool(location="北京") if tool else {"status": "error"}
        after_reconnect = default_manager.status()["servers"].get("hospital_locator", {})
        reconnect_ok = (
            recovered.get("status") == "ok"
            and recovered["data"].get("count", 0) >= 1
            and after_reconnect.get("reconnects", 0) >= 1
            and after_reconnect.get("connected") is True
        )

        # 科室过滤无匹配时返回空列表（演示源）；高德源无科室信息不过滤、返回非空
        no_dept = tool(location="北京", department="儿科") if tool else {"data": {}}
        no_dept_note = (no_dept.get("data") or {}).get("note", "")
        dept_mismatch_ok = no_dept.get("status") == "ok" and (
            ("高德" in no_dept_note and no_dept["data"].get("count", 0) >= 1)
            or no_dept["data"].get("count", 0) == 0
        )

        # HIGH 升级回复：配置位置后附加医院清单；未配置时不附加（保持既有回复形态）
        bare = build_risk_escalation_action_result(
            {"symptoms": ["胸痛"]},
            {"risk": "HIGH", "reason": "测试原因", "disposition": "建议就医"},
            {"advice": []},
            {"syndrome_candidates": []},
        )
        os.environ["MEDICAL_AGENT_LOCATION"] = "北京"
        enriched = build_risk_escalation_action_result(
            {"symptoms": ["胸痛"]},
            {"risk": "HIGH", "reason": "测试原因", "disposition": "建议就医"},
            {"advice": []},
            {"syndrome_candidates": []},
        )
        hospital_lines_ok = (
            "医院可前往" not in bare.response
            and "医院可前往" in enriched.response
            and ("高德" in enriched.response or "演示数据" in enriched.response)
        )

        # city 槽位优先级链：会话抽取的城市优先于 env；无位置来源时不附加
        os.environ.pop("MEDICAL_AGENT_LOCATION", None)
        city_driven = build_risk_escalation_action_result(
            {"symptoms": ["胸痛"], "city": "上海"},
            {"risk": "HIGH", "reason": "测试原因", "disposition": "建议就医"},
            {"advice": []},
            {"syndrome_candidates": []},
        )
        os.environ["MEDICAL_AGENT_LOCATION"] = "北京"
        city_over_env = build_risk_escalation_action_result(
            {"symptoms": ["胸痛"], "city": "广州"},
            {"risk": "HIGH", "reason": "测试原因", "disposition": "建议就医"},
            {"advice": []},
            {"syndrome_candidates": []},
        )
        city_slot_ok = (
            "就近位于上海" in city_driven.response
            and "就近位于广州" in city_over_env.response
        )

        # C6：医院数据源外置——未配置远端走演示数据；配置了不可达远端时回退演示数据并在 note 标注
        from mcp_servers.hospital_locator import _load_hospitals
        from mcp_servers.hospital_locator import search_nearby_hospitals as raw_search

        demo_hospitals, demo_source = _load_hospitals()
        os.environ["HOSPITAL_DATA_URL"] = "http://127.0.0.1:9/hospitals.json"
        try:
            fallback_payload = json.loads(raw_search("北京"))
        finally:
            os.environ.pop("HOSPITAL_DATA_URL", None)
        data_source_switch_ok = (
            demo_source == "demo"
            and bool(demo_hospitals)
            and fallback_payload.get("count", 0) > 0
            and "已回退演示数据" in fallback_payload.get("note", "")
        )

        # C7：高德地图数据源——解析函数离线验证；仅配 key 但服务不可达时回退演示数据并在 note 标注
        from mcp_servers.hospital_locator import _parse_amap_pois

        amap_payload = {
            "status": "1",
            "pois": [
                {"name": "示例人民医院", "adname": "示例区"},
                {"noname": True},
            ],
        }
        parsed_pois = _parse_amap_pois(amap_payload)
        amap_error_rejected = False
        try:
            _parse_amap_pois({"status": "0"})
        except ValueError:
            amap_error_rejected = True
        amap_parse_ok = (
            len(parsed_pois) == 1
            and parsed_pois[0]["name"] == "示例人民医院"
            and parsed_pois[0]["district"] == "示例区"
            and parsed_pois[0]["departments"] == []
            and amap_error_rejected
        )

        os.environ["AMAP_API_KEY"] = "fake-key"
        os.environ["AMAP_BASE_URL"] = "http://127.0.0.1:9"
        try:
            amap_fallback_payload = json.loads(raw_search("北京"))
        finally:
            os.environ.pop("AMAP_API_KEY", None)
            os.environ.pop("AMAP_BASE_URL", None)
        amap_fallback_ok = (
            amap_fallback_payload.get("count", 0) > 0
            and "高德地图服务不可用" in amap_fallback_payload.get("note", "")
        )

        # 数据源优先级：同时配置静态 URL 与高德 key 时走 URL 分支（note 为静态源回退措辞而非高德）
        os.environ["AMAP_API_KEY"] = "fake-key"
        os.environ["HOSPITAL_DATA_URL"] = "http://127.0.0.1:9/hospitals.json"
        try:
            url_priority_payload = json.loads(raw_search("北京"))
        finally:
            os.environ.pop("AMAP_API_KEY", None)
            os.environ.pop("HOSPITAL_DATA_URL", None)
        url_priority_ok = "外部数据源不可用" in url_priority_payload.get("note", "")

        # 周边搜索（坐标模式）：距离换算、10km 空自动扩 20km 取最近 5 家、失败无城市诚实返空、升级链路坐标渲染
        import mcp_servers.hospital_locator as hl
        from mcp_servers.hospital_locator import _parse_amap_around_pois

        around_entries = _parse_amap_around_pois({
            "status": "1",
            "pois": [
                {"name": "就近人民医院", "adname": "示例区", "distance": "1234"},
                {"name": "无距离人民医院", "adname": "示例区", "distance": ""},
            ],
        })
        around_parse_ok = (
            len(around_entries) == 2
            and around_entries[0]["distance_km"] == 1.2
            and "distance_km" not in around_entries[1]
            and "distance_raw" not in around_entries[0]
        )

        original_around = hl._search_amap_around
        around_calls = []

        def fake_around_empty_then_hit(latitude, longitude, api_key, radius_meters):
            around_calls.append(radius_meters)
            if radius_meters == hl.AMAP_NEARBY_RADIUS_METERS:
                return []
            return [
                {"name": f"就近医院{i}", "district": "示例区", "departments": [], "has_emergency": False, "distance_km": 11.0 + i}
                for i in range(7)
            ]

        os.environ["AMAP_API_KEY"] = "fake-key"
        os.environ.pop("HOSPITAL_DATA_URL", None)
        hl._search_amap_around = fake_around_empty_then_hit
        try:
            nearby_expanded = json.loads(raw_search(location="", latitude=31.23, longitude=121.47))
        finally:
            hl._search_amap_around = original_around
        nearby_expand_ok = (
            around_calls == [hl.AMAP_NEARBY_RADIUS_METERS, hl.AMAP_NEARBY_RADIUS_EXPANDED_METERS]
            and nearby_expanded.get("source") == "amap_nearby"
            and nearby_expanded.get("radius_km") == 20
            and nearby_expanded.get("count") == 5
            and "已扩大至 20km" in nearby_expanded.get("note", "")
        )

        around_calls.clear()
        hl._search_amap_around = lambda latitude, longitude, api_key, radius_meters: (
            around_calls.append(radius_meters)
            or [{"name": "就近医院", "district": "示例区", "departments": [], "has_emergency": False, "distance_km": 0.8}]
        )
        try:
            nearby_hit = json.loads(raw_search(location="", latitude=31.23, longitude=121.47))
        finally:
            hl._search_amap_around = original_around
        nearby_hit_ok = (
            around_calls == [hl.AMAP_NEARBY_RADIUS_METERS]
            and nearby_hit.get("radius_km") == 10
            and "已扩大" not in nearby_hit.get("note", "")
        )

        # 坐标检索不可用且无城市可回退：诚实返回空，不用无关城市演示数据冒充就近
        os.environ["AMAP_BASE_URL"] = "http://127.0.0.1:9"
        try:
            coords_fail = json.loads(raw_search(location="", latitude=31.23, longitude=121.47))
        finally:
            os.environ.pop("AMAP_BASE_URL", None)
            os.environ.pop("AMAP_API_KEY", None)
        coords_fail_ok = (
            coords_fail.get("count", 0) == 0
            and "周边检索不可用" in coords_fail.get("note", "")
        )

        # 升级链路坐标模式渲染：抬头带半径说明、取最近 5 家，且与城市模式共用兜底标记前缀
        from agent.runtime_utils import _nearby_hospital_lines
        from tools.registry import default_registry

        def fake_nearby_tool(**kwargs):
            if kwargs.get("latitude") and kwargs.get("longitude"):
                return {
                    "status": "ok",
                    "data": {
                        "hospitals": [
                            {"name": f"就近医院{i}", "district": "示例区", "distance_km": round(0.5 * (i + 1), 1)}
                            for i in range(6)
                        ],
                        "source": "amap_nearby",
                        "radius_km": 10,
                        "note": "数据来自高德地图 POI（周边检索，按直线距离排序），就医请以实际导航与医院公告为准",
                    },
                }
            return {"status": "ok", "data": {"hospitals": [], "source": "demo", "note": ""}}

        original_registry_get = default_registry.get
        default_registry.get = lambda name: (
            fake_nearby_tool if name == "hospital_locator_search_nearby_hospitals" else original_registry_get(name)
        )
        try:
            coords_lines = _nearby_hospital_lines({"user_coords": {"latitude": 31.23, "longitude": 121.47}})
        finally:
            default_registry.get = original_registry_get
        coords_render_ok = (
            len(coords_lines) >= 7
            and coords_lines[0].startswith("如你就近位于当前位置，10km 内以下医院可前往")
            and sum(1 for line in coords_lines if line.startswith("- ")) == 5
            and "约 0.5 公里" in coords_lines[1]
        )
    finally:
        os.environ.pop("MEDICAL_AGENT_LOCATION", None)
        os.environ.pop("HOSPITAL_DATA_URL", None)
        os.environ.pop("AMAP_API_KEY", None)
        os.environ.pop("AMAP_BASE_URL", None)
        shutdown_mcp_servers()

    checks = [
        ("mock mcp server connected via stdio", connected),
        ("status exposes connection state and counters", status_ok),
        ("search_nearby_hospitals returns structured hospitals", data_ok),
        ("unknown location returns empty list without error", unknown_ok),
        ("unmatched department returns empty instead of silent fallback", dept_mismatch_ok),
        ("call metrics counted without errors", metrics_ok),
        ("dead session auto-reconnects and recovers", reconnect_ok),
        ("high-risk escalation appends hospitals only when configured", hospital_lines_ok),
        ("conversation city slot drives hospital list over env", city_slot_ok),
        ("hospital data source falls back to demo when remote unreachable", data_source_switch_ok),
        ("amap poi payload parses into honest entries", amap_parse_ok),
        ("amap unreachable falls back to demo with note", amap_fallback_ok),
        ("static url takes priority over amap source", url_priority_ok),
        ("amap around payload converts distance meters to km", around_parse_ok),
        ("nearby search expands radius to 20km when 10km empty", nearby_expand_ok),
        ("nearby search keeps 10km and returns closest five when hit", nearby_hit_ok),
        ("nearby search failure without city returns honest empty", coords_fail_ok),
        ("escalation renders coords mode header with five hospitals", coords_render_ok),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试红旗否定句豁免：否认表述不误报红旗，后置否定与非否定命中不受影响
def test_red_flag_negation_exemption():
    from tools.symptom_tool import extract_symptoms

    denied = extract_symptoms("我没有喘不过气，也没有便血，就是有点乏力")
    affirmed = extract_symptoms("胸痛不缓解，已经两个小时了")
    mixed = extract_symptoms("没有便血，但今天确实便血了一次")

    checks = [
        ("negated red flags not reported", not denied["red_flags"]),
        ("negated symptoms still extracted", "乏力" in denied["symptoms"]),
        ("affirmed red flag still reported", "持续胸痛" in affirmed["red_flags"]),
        ("non-negated occurrence wins", "便血" in mixed["red_flags"]),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试城市槽位：机会式抽取、覆盖式合并进 case_state，不进追问槽位表
def test_city_slot_extraction_and_merge():
    from knowledge.tcm_knowledge import TCM_SLOT_LABELS
    from memory.memory import ConversationMemory
    from tools.symptom_tool import extract_symptoms

    with_city = extract_symptoms("我在上海，这两天咳嗽怕冷，有点发热")
    without_city = extract_symptoms("这两天咳嗽怕冷，有点发热")
    hangzhou_city = extract_symptoms("我在杭州，突然胸口闷疼")

    memory = ConversationMemory()
    memory.update_case(with_city)
    first_city = memory.get_case_state().get("city")
    memory.update_case({"city": "北京"})
    state = memory.get_case_state()
    memory.update_user_coords(31.23, 121.47)
    coords_state = memory.get_case_state()

    checks = [
        ("city extracted from free text", with_city["city"] == "上海"),
        ("expanded city vocab covers hangzhou", hangzhou_city["city"] == "杭州"),
        ("text without city yields empty slot", without_city["city"] == ""),
        ("city merged into case_state", first_city == "上海"),
        ("latest city statement overwrites previous", state["city"] == "北京"),
        ("city stays out of triage slot labels", "city" not in TCM_SLOT_LABELS),
        ("browser coords stored in session case state", coords_state.get("user_coords") == {"latitude": 31.23, "longitude": 121.47}),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试 classic 运行时的矛盾消除对齐：澄清回复后以最新值消除矛盾（与 LangGraph 路径一致）
def test_classic_runtime_resolves_contradiction():
    import agent.controller as controller_module
    from agent.controller import MedicalAgent
    from llm.llm import LLM
    from memory.memory import ConversationMemory

    memory = ConversationMemory()
    memory.update_case({"age": "25", "symptoms": ["腹痛"]})
    memory.update_case({"age": "65"})
    memory.update_triage(last_action="clarify_conflict", followup_slot="age")
    had_contradiction = bool(memory.get_case_state()["contradictions"])

    original_extract = controller_module.extract_case_slots
    controller_module.extract_case_slots = lambda llm, user_input: {"age": "65"}
    try:
        agent = MedicalAgent(LLM(provider="mock"), memory)
        agent.run("我确实是65岁，没说错")
    finally:
        controller_module.extract_case_slots = original_extract

    state = memory.get_case_state()
    checks = [
        ("contradiction existed before reconfirmation", had_contradiction),
        ("classic runtime resolves contradiction after clarify reply", not state["contradictions"]),
        ("latest age kept in slot_history", state["slot_history"].get("age") == ["65"]),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试 SessionCache：同 key 复用实例、TTL 过期驱逐、超限按最旧淘汰
def test_session_cache_lifecycle():
    from app import SessionCache

    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

    clock = FakeClock()
    created = {}

    def make_factory(key):
        def factory():
            created[key] = created.get(key, 0) + 1
            return {"id": key}

        return factory

    cache = SessionCache(ttl_seconds=100, max_entries=2, clock=clock)
    first = cache.get_or_create("a", make_factory("a"))
    same = cache.get_or_create("a", make_factory("a"))

    clock.now = 101
    renewed = cache.get_or_create("a", make_factory("a"))

    lru_cache = SessionCache(ttl_seconds=100, max_entries=2, clock=clock)
    clock.now = 200
    lru_cache.get_or_create("x", make_factory("x"))
    clock.now = 201
    lru_cache.get_or_create("y", make_factory("y"))
    clock.now = 202
    lru_cache.get_or_create("x", make_factory("x"))
    clock.now = 203
    lru_cache.get_or_create("z", make_factory("z"))

    checks = [
        ("same session returns same agent instance", first["agent"] is same["agent"]),
        ("expired entry is recreated", renewed["agent"] is not first["agent"]),
        ("lru evicts least recently used", len(lru_cache) == 2),
        ("recently accessed entry survives eviction", created.get("x") == 1 and created.get("z") == 1),
        ("oldest untouched entry is evicted", created.get("y") == 1),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试 JSON 原子写：写入后无临时文件残留，损坏的目标文件可被正确覆盖
def test_atomic_json_write():
    import tempfile

    from memory.file_store import read_json_file, write_json_file

    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, "session.json")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("{broken")

        write_json_file(target, {"case_state": {"age": "25"}})
        data = read_json_file(target)
        tmp_leftover = os.path.exists(target + ".tmp")

    checks = [
        ("corrupted file replaced with valid json", data == {"case_state": {"age": "25"}}),
        ("no tmp file left behind", not tmp_leftover),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试 LLM 重试与降级透明化：网络异常重试一次后降级 mock，状态字段如实暴露
def test_llm_retry_and_degradation_transparency():
    import llm.llm as llm_module
    from llm.llm import LLM

    class FakeRequestError(Exception):
        pass

    class FakeRequests:
        RequestException = FakeRequestError

        def __init__(self):
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            raise FakeRequestError("network down")

    fake = FakeRequests()
    original_requests = llm_module.requests
    llm_module.requests = fake
    llm = LLM(provider="deepseek")
    original_url = os.environ.get("DEEPSEEK_API_URL")
    original_key = os.environ.get("DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_URL"] = "http://example.invalid/v1/chat/completions"
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    try:
        result = llm.call("你好")
    finally:
        llm_module.requests = original_requests
        if original_url is None:
            os.environ.pop("DEEPSEEK_API_URL", None)
        else:
            os.environ["DEEPSEEK_API_URL"] = original_url
        if original_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = original_key

    status = llm.get_runtime_status()
    checks = [
        ("network failure retried once", fake.calls == 2),
        ("fell back to mock after retries", llm.last_provider_used == "mock" and bool(result)),
        ("degraded flag exposed in status", status["degraded"] is True),
        ("fallback timestamp recorded", bool(status["last_fallback_at"])),
        ("last_error recorded", bool(status["last_error"])),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试风险规则外置：声明式规则加载校验通过，解释器对典型规则命中正确
def test_risk_rules_externalized():
    from knowledge.tcm_knowledge import KNOWLEDGE_VERSION, get_risk_rules
    from tools.risk_tool import risk_assessment

    rules = get_risk_rules()
    rules_valid = (
        len(rules) >= 10
        and all(rule.get("id") and rule.get("when") and rule.get("reason") for rule in rules)
        and "risk_rules" in KNOWLEDGE_VERSION
    )

    combo = risk_assessment({"symptoms": ["胸痛", "呼吸困难"]})
    cardiac = risk_assessment({"symptoms": ["胸痛"], "past_history": ["高血压"]})
    headache = risk_assessment({"symptoms": ["头痛", "呕吐"], "severity": "中度"})
    elderly = risk_assessment({"symptoms": ["乏力"], "age": "70"})

    checks = [
        ("risk rules loaded with schema validation", rules_valid),
        ("symptom combo rule hits", combo["risk"] == "HIGH" and "chest_pain_with_dyspnea" in combo["matched_rules"]),
        (
            "past history condition hits",
            cardiac["risk"] == "HIGH" and "chest_pain_with_cardiac_history" in cardiac["matched_rules"],
        ),
        (
            "severity plus union condition hits",
            headache["risk"] == "MEDIUM" and "headache_with_vomiting" in headache["matched_rules"],
        ),
        ("age condition hits", elderly["risk"] == "MEDIUM" and "older_patient" in elderly["matched_rules"]),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试配置校验友好化：非数字参数抛出含字段名的明确错误
def test_config_validation_friendly_errors():
    from config_manager import _validate_config

    def expect_field_error(config, field):
        try:
            _validate_config(config)
            return False
        except ValueError as exc:
            return field in str(exc)

    checks = [
        (
            "non-numeric max_tokens raises friendly error",
            expect_field_error(
                {"LLM_PROVIDER": "deepseek", "DEEPSEEK_MAX_TOKENS": "abc", "DEEPSEEK_TEMPERATURE": "0.2"},
                "DEEPSEEK_MAX_TOKENS",
            ),
        ),
        (
            "non-numeric temperature raises friendly error",
            expect_field_error(
                {"LLM_PROVIDER": "deepseek", "DEEPSEEK_MAX_TOKENS": "512", "DEEPSEEK_TEMPERATURE": "abc"},
                "DEEPSEEK_TEMPERATURE",
            ),
        ),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试配置序列化保留手填可选键：配置页保存不得清掉 AMAP_API_KEY 等（防数据源静默降级）
def test_config_save_preserves_manual_extra_keys():
    from config_manager import DEFAULT_CONFIG, _serialize_config

    config = dict(DEFAULT_CONFIG)
    config["AMAP_API_KEY"] = "k" * 32
    config["HOSPITAL_DATA_URL"] = "http://example.invalid/hospitals.json"
    text = _serialize_config(config)

    checks = [
        ("serialize keeps amap key", f"AMAP_API_KEY={'k' * 32}" in text),
        ("serialize keeps hospital url", "HOSPITAL_DATA_URL=http://example.invalid/hospitals.json" in text),
        ("serialize keeps default keys", "LLM_PROVIDER=" in text),
        ("serialize drops blank extras", "EMPTY_KEY" not in _serialize_config({**DEFAULT_CONFIG, "EMPTY_KEY": "  "})),
    ]

    passed = True
    for label, ok in checks:
        if ok:
            print(f"[PASS] {label}")
        else:
            passed = False
            print(f"[FAIL] {label}")
    return passed


# 测试医院清单的 LLM 重写兜底：重写丢失时机械补回，已含或草稿无清单时原样返回
def test_hospital_lines_rewrite_fallback():
    from agent.runtime_utils import _ensure_hospital_lines

    draft = (
        "分诊建议正文。\n"
        "如你就近位于上海，以下医院可前往：\n"
        "- 复旦大学附属华山医院（静安区）\n"
        "（数据来自高德地图 POI，就医请以实际导航与医院公告为准）"
    )
    restored = _ensure_hospital_lines("改写后丢了医院清单的回复", draft)
    untouched = _ensure_hospital_lines("回复里已含如你就近位于段落", draft)

    checks = [
        ("hospital fallback restores dropped list", "复旦大学附属华山医院" in restored and restored.endswith("医院公告为准）")),
        ("hospital fallback no-op when present", untouched == "回复里已含如你就近位于段落"),
        ("hospital fallback no-op without draft list", _ensure_hospital_lines("任意回复", "无医院段的草稿") == "任意回复"),
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
        test_manual_profile_save_and_prefill(),
        test_profile_api_roundtrip(),
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
        test_knowledge_data_loading(),
        test_knowledge_retrieval_tool(),
        test_planner_rule_enhancements(),
        test_planner_review_interceptions(),
        test_evaluation_case_smoke(),
        test_contradiction_resolution(),
        test_planner_no_empty_bundle(),
        test_red_flag_negation_exemption(),
        test_city_slot_extraction_and_merge(),
        test_classic_runtime_resolves_contradiction(),
        test_session_cache_lifecycle(),
        test_atomic_json_write(),
        test_llm_retry_and_degradation_transparency(),
        test_config_validation_friendly_errors(),
        test_risk_rules_externalized(),
        test_corpus_knowledge_and_retrieval(),
        test_mcp_config_and_adapter_degradation(),
        test_mcp_end_to_end_hospital_locator(),
        test_hospital_lines_rewrite_fallback(),
        test_config_save_preserves_manual_extra_keys(),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\nSummary: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
