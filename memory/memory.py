# 构建整个Agent的记忆系统基础设施
# 身份+安全+长短期记忆分层

# 生成唯一标识符
# user_id  session_id  thread_id
# 如何区分不同用户/不同问诊会话
# 1. user_id：代表一个独立的用户，可以跨多个会话保持一致。适用于需要长期画像和跨会话记忆的场景。
# 2. session_id：代表一次独立的问诊会话，每次新的问诊都会生成一个新的session_id。适用于每次问诊都需要独立记忆的场景。
# 3. thread_id：如果需要在同一会话中区分不同的对话线程（例如多轮问诊中的不同阶段），可以使用thread_id来标识不同的对话线程。
import uuid
# 深拷贝，完全复制所有嵌套结构
# 保证memory是不可被外部污染的真实单一数据源
from copy import deepcopy
# 长期用户画像存储系统
# 年龄、性别、慢性病史、过敏史、用药史
# 多次问诊累计信息、个性化诊断、长期健康建模
from memory.profile_store import InMemoryProfileStore
# 当前会话状态存储系统
# 对话历史、当前case_state
from memory.session_store import InMemorySessionStore

# 存数据、更新数据、给LLM提供上下文
# 数字病人状态机 + 记忆数据库 + 上下文压缩器
class ConversationMemory:
    # 长期稳定信息（长期记忆）
    # 列表型，可累积
    # 不会因为一次问诊改变，会长期影响诊断，需要跨session保留
    PROFILE_LIST_FIELDS = [
        "past_history",
        "allergy_history",
        "medication_history",
    ]

    # 长期信息+标量（只能有一个值）
    # 不会累积，只会覆盖
    PROFILE_SCALAR_FIELDS = [
        "age",
        "sex",
    ]

    # 当前问诊中可累积的信息
    # 例如：发烧、头痛等症状可能在问诊过程中被多次提及
    # 每次提及都可以累积到列表中，形成完整的症状集合
    LIST_FIELDS = [
        "symptoms",
        "accompanying_symptoms",
        "past_history",
        "allergy_history",
        "medication_history",
        "red_flags",
        "pulse_candidates",
        "syndrome_candidates",
    ]

    # 当前问诊中会被覆盖的单值字段
    # 例如：3天改成5天，轻度改成重度
    SCALAR_FIELDS = [
        "chief_complaint",
        "duration",
        "severity",
        "location",
        "age",
        "sex",
        "description",
        "cold_heat",
        "sweating",
        "thirst",
        "appetite",
        "sleep",
        "stool_urine",
        "pain_character",
        "emotion",
        "complexion",
        "voice_breath",
        "odor",
        "female_cycle",
        "pulse_summary",
        "pulse_strength",
        "pulse_rate",
        "pulse_source",
        "pulse_signal_quality",
    ]

    # 构造函数，初始化记忆系统
    def __init__(
        self,
        max_history=12,
        profile_store=None,
        user_id=None,
        session_store=None,
        session_id=None,
    ):
        # max_history: 最多保存多少轮对话
        self.max_history = max_history
        # prompt_history_window: 给LLM用的窗口大小
        # 通常比max_history小，保证提供最新的上下文
        self.prompt_history_window = 4
        # user_id和session_id的生成逻辑
        self.user_id = user_id or str(uuid.uuid4())
        self.session_id = session_id or str(uuid.uuid4())
        # memory分成两层：profile长期和session短期
        self.profile_store = profile_store or InMemoryProfileStore()
        self.session_store = session_store or InMemorySessionStore()
        # 读取历史，如果有旧会话，就恢复
        self.history = self.session_store.get_history(self.session_id, [])
        # 初始化用户画像，如果没有，就创建默认
        self.profile_store.get_profile(self.user_id, self._default_profile())
        # 初始化case_state，包含当前问诊相关的所有信息
        # 数字病人
        # 医学知识图谱、Agent状态机、推理缓存、LLM上下文
        self.case_state = self.session_store.get_case_state(self.session_id, {
            "chief_complaint": "",
            "symptoms": [],
            "accompanying_symptoms": [],
            "duration": "",
            "severity": "",
            "location": "",
            "age": "",
            "sex": "",
            "past_history": [],
            "allergy_history": [],
            "medication_history": [],
            "red_flags": [],
            "cold_heat": "",
            "sweating": "",
            "thirst": "",
            "appetite": "",
            "sleep": "",
            "stool_urine": "",
            "pain_character": "",
            "emotion": "",
            "complexion": "",
            "voice_breath": "",
            "odor": "",
            "female_cycle": "",
            "pulse_summary": "",
            "pulse_strength": "",
            "pulse_rate": "",
            "pulse_source": "",
            "pulse_signal_quality": "",
            "pulse_candidates": [],
            "pulse_prompt_count": 0,
            "pulse_declined": False,
            "syndrome_candidates": [],
            "risk_level": "",
            "status": "INIT",
            "missing_slots": [],
            "last_action": "",
            "last_followup_slot": "",
            "followup_counts": {},
            "action_history": [],
            "goal": "完成安全分诊与中医问诊信息收集",
            "goal_progress": "等待首轮信息输入",
            "pending_questions": [],
            "open_questions": [],
            "hypotheses": [],
            "confidence": 0.0,
            "contradictions": [],
            "contradiction_fields": [],
            "slot_history": {},
            "self_check": {},
            "stop_reason": "",
            "internal_steps": 0,
            "resolved_facts": [],
            "conversation_summary": "当前还缺少足够的问诊上下文。",
            "summary": "",
            "tcm_summary": "",
            "description": "",
        })

    # 添加用户输入到历史中，并更新相关状态
    # 1. 将用户输入添加到history列表中，记录角色和文本
    # 2. 限制长度，只保留最近的max_history轮对话
    # 3. 更新摘要
    # 4. 持久化
    def add_user(self, text):
        self.history.append(("user", text))
        self._trim_history()
        self._refresh_conversation_memory()
        self._persist_session_state()

    def add_assistant(self, text):
        self.history.append(("assistant", text))
        self._trim_history()
        self._refresh_conversation_memory()
        self._persist_session_state()

    # 核心更新函数
    def update_case(self, extracted_slots):
        # LIST字段处理
        # 旧+新，合并去重，累积信息
        for field in self.LIST_FIELDS:
            merged = self.case_state.get(field, []) + extracted_slots.get(field, [])
            self.case_state[field] = list(dict.fromkeys(item for item in merged if item))

        # SCALAR字段处理
        # 直接覆盖，记录用户前后说过什么，方便矛盾检测
        for field in self.SCALAR_FIELDS:
            if extracted_slots.get(field):
                self._record_slot_history(field, extracted_slots[field])
                self.case_state[field] = extracted_slots[field]

        # 长期信息提升：把稳定信息写入profile
        # 形成长期画像，支持跨会话记忆和个性化
        self._promote_to_long_term_profile(extracted_slots)
        self._refresh_contradictions()
        # 更新摘要，提供给LLM用的上下文信息
        self.case_state["summary"] = self._build_summary()
        self.case_state["tcm_summary"] = self._build_tcm_summary()
        self._refresh_conversation_memory()
        self._persist_session_state()

    # 专门处理脉搏数据的输入
    def update_pulse_data(self, pulse_data):
        normalized = {
            "pulse_summary": pulse_data.get("pulse_summary", ""),
            "pulse_strength": pulse_data.get("pulse_strength", ""),
            "pulse_rate": pulse_data.get("pulse_rate", ""),
            # 区分脉搏数据的来源，方便后续分析和处理
            "pulse_source": pulse_data.get("pulse_source", "pulse_device"),
            "pulse_signal_quality": pulse_data.get("pulse_signal_quality", ""),
            "pulse_candidates": pulse_data.get("pulse_candidates", []),
        }
        self.case_state["pulse_declined"] = False
        self.update_case(normalized)

    # 更新风险评估和分诊状态
    # 控制问诊流程
    def update_triage(
        self,
        risk_result=None,
        status=None,
        last_action=None,
        missing_slots=None,
        followup_slot=None,
        syndrome_candidates=None,
    ):
        if risk_result and risk_result.get("risk"):
            self.case_state["risk_level"] = risk_result["risk"]
        if status:
            self.case_state["status"] = status
        if last_action:
            self.case_state["last_action"] = last_action
        if missing_slots is not None:
            self.case_state["missing_slots"] = missing_slots
        if syndrome_candidates is not None:
            self.case_state["syndrome_candidates"] = syndrome_candidates
        if followup_slot:
            counts = dict(self.case_state.get("followup_counts", {}))
            counts[followup_slot] = counts.get(followup_slot, 0) + 1
            self.case_state["followup_counts"] = counts
            self.case_state["last_followup_slot"] = followup_slot
        self._persist_session_state()

    # Agent认知状态
    # 当前进展、未解决问题、假设、置信度
    # Agent的思考状态
    def update_task_state(
        self,
        goal_progress=None,
        pending_questions=None,
        hypotheses=None,
        confidence=None,
        contradictions=None,
        contradiction_fields=None,
        self_check=None,
        stop_reason=None,
        internal_steps=None,
    ):
        if goal_progress:
            self.case_state["goal_progress"] = goal_progress
        if pending_questions is not None:
            self.case_state["pending_questions"] = pending_questions
            self.case_state["open_questions"] = list(pending_questions)
        if hypotheses is not None:
            self.case_state["hypotheses"] = hypotheses
        if confidence is not None:
            self.case_state["confidence"] = confidence
        if contradictions is not None:
            self.case_state["contradictions"] = contradictions
        if contradiction_fields is not None:
            self.case_state["contradiction_fields"] = contradiction_fields
        if self_check is not None:
            self.case_state["self_check"] = self_check
        if stop_reason is not None:
            self.case_state["stop_reason"] = stop_reason
        if internal_steps is not None:
            self.case_state["internal_steps"] = internal_steps
        self._refresh_conversation_memory()
        self._persist_session_state()

    # 行为日志，记录每一步干了什么
    def record_action(self, action_name, status, step, reason="", missing_slots=None):
        history = list(self.case_state.get("action_history", []))
        history.append(
            {
                "step": step,
                "action": action_name,
                "status": status,
                "reason": reason,
                "missing_slots": list(missing_slots or []),
            }
        )
        self.case_state["action_history"] = history[-8:]
        if action_name == "request_pulse_input":
            self.case_state["pulse_prompt_count"] = self.case_state.get("pulse_prompt_count", 0) + 1
        self._persist_session_state()

    # 用户拒绝提供脉搏数据，记录这个事件，可能影响后续的问诊策略
    def mark_pulse_skipped(self):
        self.case_state["pulse_declined"] = True
        self._persist_session_state()

    # 获取当前case_state的深拷贝，确保外部无法直接修改内部状态
    def get_case_state(self):
        return deepcopy(self.case_state)

    # 获取长期画像，供LLM用来提供个性化建议和诊断
    def get_long_term_profile(self):
        return self.profile_store.get_profile(self.user_id, self._default_profile())

    # 获取当前对话上下文，供LLM用来理解当前问诊进展和历史信息
    def get_context(self):
        return self.history[-self.max_history :]

    # 将上下文格式化成文本，供LLM用来提供诊断建议和决策支持
    def get_context_text(self):
        return "\n".join(f"{role}: {text}" for role, text in self.get_context())

    # 获取长期画像的文本描述，供LLM用来提供个性化建议和诊断
    def get_profile_context_text(self):
        return self.get_long_term_profile().get("profile_summary", "长期画像暂未形成")

    # summary+最近对话
    # summary提供了当前问诊的整体进展和关键信息
    # 最近对话提供了最新的上下文细节
    def get_prompt_context_text(self):
        summary = self.case_state.get("conversation_summary", "当前还缺少足够的问诊上下文。")
        recent_history = self.history[-self.prompt_history_window :]
        recent_text = "\n".join(f"{role}: {text}" for role, text in recent_history)
        if recent_text:
            return f"summary: {summary}\nrecent_history:\n{recent_text}"
        return f"summary: {summary}"

    # 限制历史长度，避免内存无限增长，同时保证提供足够的上下文给LLM
    def _trim_history(self):
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    # 西医摘要
    def _build_summary(self):
        parts = []

        if self.case_state["symptoms"]:
            parts.append(f"症状: {'、'.join(self.case_state['symptoms'])}")
        if self.case_state["accompanying_symptoms"]:
            parts.append(f"伴随症状: {'、'.join(self.case_state['accompanying_symptoms'])}")
        if self.case_state["location"]:
            parts.append(f"部位: {self.case_state['location']}")
        if self.case_state["duration"]:
            parts.append(f"病程: {self.case_state['duration']}")
        if self.case_state["severity"]:
            parts.append(f"程度: {self.case_state['severity']}")
        if self.case_state["past_history"]:
            parts.append(f"既往史: {'、'.join(self.case_state['past_history'])}")
        if self.case_state["red_flags"]:
            parts.append(f"红旗信号: {'、'.join(self.case_state['red_flags'])}")

        if not parts:
            return "等待补充症状信息"
        return "；".join(parts)

    # 中医摘要
    def _build_tcm_summary(self):
        parts = []
        if self.case_state["cold_heat"]:
            parts.append(f"寒热: {self.case_state['cold_heat']}")
        if self.case_state["sweating"]:
            parts.append(f"汗出: {self.case_state['sweating']}")
        if self.case_state["thirst"]:
            parts.append(f"口渴: {self.case_state['thirst']}")
        if self.case_state["appetite"]:
            parts.append(f"纳食: {self.case_state['appetite']}")
        if self.case_state["sleep"]:
            parts.append(f"睡眠: {self.case_state['sleep']}")
        if self.case_state["stool_urine"]:
            parts.append(f"二便: {self.case_state['stool_urine']}")
        if self.case_state["pain_character"]:
            parts.append(f"疼痛性质: {self.case_state['pain_character']}")
        if self.case_state["emotion"]:
            parts.append(f"情志: {self.case_state['emotion']}")
        if self.case_state["complexion"]:
            parts.append(f"面色神情: {self.case_state['complexion']}")
        if self.case_state["voice_breath"]:
            parts.append(f"声音气息: {self.case_state['voice_breath']}")
        if self.case_state["pulse_summary"]:
            pulse_text = self.case_state["pulse_summary"]
            if self.case_state["pulse_signal_quality"]:
                pulse_text += f"(信号{self.case_state['pulse_signal_quality']})"
            parts.append(f"脉诊: {pulse_text}")

        if not parts:
            return "问诊证据仍不足"
        return "；".join(parts)

    # 记录标量字段的历史值，方便后续进行矛盾检测
    def _record_slot_history(self, field, value):
        history = deepcopy(self.case_state.get("slot_history", {}))
        values = list(history.get(field, []))
        normalized = str(value).strip()
        if normalized and normalized not in values:
            values.append(normalized)
        history[field] = values
        self.case_state["slot_history"] = history

    # 检查标量字段的历史值，记录矛盾
    # 检测信息冲突
    def _refresh_contradictions(self):
        contradiction_templates = {
            "age": "年龄信息前后不一致",
            "sex": "性别信息前后不一致",
            "severity": "严重程度描述前后不一致",
        }

        slot_history = self.case_state.get("slot_history", {})
        contradiction_fields = []
        contradictions = []
        for field, message in contradiction_templates.items():
            values = slot_history.get(field, [])
            if len(values) > 1:
                contradiction_fields.append(field)
                contradictions.append(f"{message}: {' / '.join(values)}")

        self.case_state["contradiction_fields"] = contradiction_fields
        self.case_state["contradictions"] = contradictions

    # 把稳定信息写入长期记忆
    def _promote_to_long_term_profile(self, extracted_slots):
        profile = self.get_long_term_profile()

        for field in self.PROFILE_LIST_FIELDS:
            merged = profile.get(field, []) + extracted_slots.get(field, [])
            profile[field] = list(dict.fromkeys(item for item in merged if item))

        for field in self.PROFILE_SCALAR_FIELDS:
            if extracted_slots.get(field):
                profile[field] = extracted_slots[field]

        profile["profile_summary"] = self._build_profile_summary(profile)
        self.profile_store.set_profile(self.user_id, profile)

    # 构建长期画像的文本描述，供LLM用来提供个性化建议和诊断
    def _build_profile_summary(self, profile):
        parts = []
        if profile["age"]:
            parts.append(f"年龄: {profile['age']}")
        if profile["sex"]:
            parts.append(f"性别: {profile['sex']}")
        if profile["past_history"]:
            parts.append(f"长期既往史: {'、'.join(profile['past_history'])}")
        if profile["allergy_history"]:
            parts.append(f"过敏史: {'、'.join(profile['allergy_history'])}")
        if profile["medication_history"]:
            parts.append(f"长期用药/近期用药: {'、'.join(profile['medication_history'])}")

        if not parts:
            return "长期画像暂未形成"
        return "；".join(parts)

    # 默认的用户画像，适用于新用户或没有足够信息的用户，提供一个基本的结构，方便后续更新和完善
    def _default_profile(self):
        return {
            "age": "",
            "sex": "",
            "past_history": [],
            "allergy_history": [],
            "medication_history": [],
            "profile_summary": "长期画像暂未形成",
        }

    # 持久化当前会话状态，包括对话历史和case_state，保证数据不丢失，并支持后续恢复和分析
    def _persist_session_state(self):
        self.session_store.set_history(self.session_id, self.history)
        self.session_store.set_case_state(self.session_id, self.case_state)

    # 核心，把复杂状态压缩成LLM能理解的一小段文本
    def _refresh_conversation_memory(self):
        resolved_facts = self._build_resolved_facts()
        self.case_state["resolved_facts"] = resolved_facts
        self.case_state["conversation_summary"] = self._build_conversation_summary(resolved_facts)

    # 构建已确认事实的列表，供LLM用来理解当前已确认的信息，形成稳定的问诊上下文
    def _build_resolved_facts(self):
        facts = []
        if self.case_state["chief_complaint"]:
            facts.append(f"主诉={self.case_state['chief_complaint']}")
        if self.case_state["symptoms"]:
            facts.append(f"症状={'、'.join(self.case_state['symptoms'])}")
        if self.case_state["duration"]:
            facts.append(f"病程={self.case_state['duration']}")
        if self.case_state["severity"]:
            facts.append(f"程度={self.case_state['severity']}")
        if self.case_state["location"]:
            facts.append(f"部位={self.case_state['location']}")
        if self.case_state["risk_level"]:
            facts.append(f"风险等级={self.case_state['risk_level']}")
        if self.case_state["past_history"]:
            facts.append(f"既往史={'、'.join(self.case_state['past_history'])}")
        if self.case_state["cold_heat"]:
            facts.append(f"寒热={self.case_state['cold_heat']}")
        if self.case_state["appetite"]:
            facts.append(f"食欲={self.case_state['appetite']}")
        if self.case_state["sleep"]:
            facts.append(f"睡眠={self.case_state['sleep']}")
        if self.case_state["stool_urine"]:
            facts.append(f"二便={self.case_state['stool_urine']}")
        return facts[:8]

    # 构建当前问诊进展的摘要，包含已确认事实、待确认问题和矛盾信息，提供给LLM用来理解当前的问诊状态和上下文信息
    def _build_conversation_summary(self, resolved_facts):
        confirmed = "；".join(resolved_facts[:5]) if resolved_facts else "尚未确认稳定事实"
        open_questions = self.case_state.get("open_questions", [])[:3]
        open_text = "、".join(open_questions) if open_questions else "暂无显式待确认问题"
        contradictions = self.case_state.get("contradictions", [])[:2]
        contradiction_text = "；".join(contradictions) if contradictions else "暂无明显信息冲突"
        progress = self.case_state.get("goal_progress", "等待首轮信息输入")
        return (
            f"当前进展：{progress}。"
            f" 已确认：{confirmed}。"
            f" 待确认：{open_text}。"
            f" 冲突信息：{contradiction_text}。"
        )
