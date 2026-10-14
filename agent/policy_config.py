"""Opt-in web/CLI role selection. Existing sessions keep their starting policy."""
import os
from pathlib import Path
from functools import lru_cache

from agent.decision_policy import DecisionPolicy
from llm.clients import ModelClient, RunBudget, load_role_configs


@lru_cache(maxsize=1)
def _process_budget():
    limit = float(os.getenv("AGENT_POLICY_MAX_COST_CNY", "10"))
    if not 0 < limit <= 100:
        raise ValueError("AGENT_POLICY_MAX_COST_CNY must be in (0,100]")
    return RunBudget(limit)


def policy_from_environment():
    planner = os.getenv("AGENT_PLANNER_MODE", "legacy").lower()
    reviewer = os.getenv("AGENT_REVIEWER_MODE", "rule").lower()
    if planner == "legacy" and reviewer == "rule":
        return None
    if planner not in {"rule", "llm"} or reviewer not in {"rule", "llm"}:
        raise ValueError("AGENT_PLANNER_MODE/AGENT_REVIEWER_MODE must be rule or llm")
    clients = {}
    if "llm" in {planner, reviewer}:
        default = Path(__file__).resolve().parents[1] / "llm/model_roles.example.yaml"
        configs = load_role_configs(os.getenv("AGENT_MODEL_ROLES_CONFIG") or default)
        for role, mode in (("planner", planner), ("reviewer", reviewer)):
            if mode == "llm":
                client_type=ModelClient
                if os.getenv('AGENT_INTERACTION_MODE')=='joint_v1':
                    from llm.prototype_clients import PrototypeClient
                    client_type=PrototypeClient
                clients[role] = client_type(configs[role], budget=_process_budget())
    if os.getenv('AGENT_INTERACTION_MODE')=='joint_v1':
        if os.getenv('AGENT_QWEN_PLAN_CONTRACT')=='basis_v1':
            from agent.qwen_grounded_policy import GroundedQwenPolicy
            return GroundedQwenPolicy(clients.get('planner'),clients.get('reviewer'),reviewer_reason_mode=os.getenv('AGENT_QWEN_REVIEW_REASON','visible'))
        from agent.qwen_interaction_policy import QwenInteractionPolicy
        return QwenInteractionPolicy(clients.get('planner'),clients.get('reviewer'))
    return DecisionPolicy(planner_client=clients.get("planner"), reviewer_client=clients.get("reviewer"),
                          name=f"planner_{planner}_reviewer_{reviewer}",
                          prompt_version=os.getenv('AGENT_POLICY_PROMPT_VERSION','v1'))


def extraction_from_environment():
    """Explicit opt-in; a session keeps its extractor until it is recreated."""
    mode = os.getenv('AGENT_EXTRACTOR_MODE', 'legacy').lower()
    if mode == 'legacy':
        return {}
    if mode != 'semantic':
        raise ValueError('AGENT_EXTRACTOR_MODE must be legacy or semantic')
    from agent.semantic_extraction import SemanticExtractor
    from agent.complaint_intent import ComplaintIntent
    default = Path(__file__).resolve().parents[1] / 'llm/semantic_roles.example.yaml'
    configs = load_role_configs(os.getenv('AGENT_MODEL_ROLES_CONFIG') or default)
    response_mode = os.getenv('AGENT_RESPONDER_MODE', 'template').lower()
    if response_mode not in {'template', 'model'}:
        raise ValueError('semantic responder must be template or model')
    budget_mode = os.getenv('AGENT_RESPONDER_BUDGET_MODE', 'conservative')
    if budget_mode not in {'conservative', 'serving_tokenizer'}:
        raise ValueError('invalid responder budget mode')
    responder_type = ModelClient
    if budget_mode == 'serving_tokenizer':
        from llm.vllm_budget import BudgetedModelClient
        responder_type = BudgetedModelClient
    intent_mode = os.getenv('AGENT_INTENT_MODE', 'off').lower()
    if intent_mode not in {'off', 'model'}:
        raise ValueError('AGENT_INTENT_MODE must be off or model')
    intent = ComplaintIntent(ModelClient(configs['intent'], budget=_process_budget()),
        prompt_version=os.getenv('AGENT_INTENT_PROMPT_VERSION', 'v1'),
        guard_version=os.getenv('AGENT_INTENT_GUARD_VERSION', 'v1'),
        choice_prompt_version=os.getenv('AGENT_INTENT_CHOICE_PROMPT_VERSION') or None,
        structured_mode=os.getenv('AGENT_INTENT_STRUCTURED_MODE', 'off')) if intent_mode == 'model' else None
    joint=os.getenv('AGENT_INTERACTION_MODE')=='joint_v1'
    if joint:
        from agent.interaction_intent import JointIntent
        from llm.prototype_clients import PrototypeClient,PrototypeResponder
        intent=JointIntent(PrototypeClient(configs['intent'],budget=_process_budget()))
        responder_type=PrototypeResponder if budget_mode=='serving_tokenizer' else PrototypeClient
    extractor = SemanticExtractor(ModelClient(configs['extractor'], budget=_process_budget()), intent=intent,
                boundary_version=os.getenv('AGENT_EXTRACTOR_BOUNDARY_VERSION', 'v1'),
                prompt_version=os.getenv('AGENT_EXTRACTOR_PROMPT_VERSION', 'locked'),
                contract_version=os.getenv('AGENT_FACT_CONTRACT_VERSION') or None,
                representation_mode=os.getenv('AGENT_REPRESENTATION_MODE', 'finite'),
                recovery_mode=os.getenv('AGENT_EXTRACTOR_RECOVERY_MODE', 'off'),
                delivery_guard=os.getenv('AGENT_DELIVERY_GROUNDING_GUARD') or None)
    safety=os.getenv('AGENT_DELIVERY_INPUT_SAFETY') or None
    if safety not in {None,'literal_breathing_safety_v1','literal_breathing_safety_v2'}:raise ValueError('unsupported input safety version')
    extractor.input_safety_version=safety
    turn_requests=os.getenv('AGENT_DELIVERY_TURN_REQUESTS') or None
    if turn_requests not in {None,'delivery_turn_requests_v1'}:raise ValueError('unsupported turn request version')
    extractor.turn_requests_version=turn_requests
    if joint:
        from llm.prototype_clients import PrototypeClient
        extractor.client=PrototypeClient(configs['extractor'],budget=_process_budget())
        extractor.turn_requests_version='joint_interaction_v1'
    projection = os.getenv('AGENT_FACT_PROJECTION_VERSION') or None
    if projection:
        from memory.runtime_fact_projection import VERSION
        from llm.fact_contract_runtime import V2
        if projection != VERSION or extractor.contract_version != V2:
            raise ValueError('unsupported runtime fact projection')
        extractor.runtime_projection_version = projection
    responder = responder_type(configs['responder'], budget=_process_budget()) if response_mode == 'model' else None
    if responder is not None:
        context = os.getenv('AGENT_RESPONDER_CONTEXT_VERSION', 'v2')
        guard = os.getenv('AGENT_RESPONDER_FACT_GUARD_VERSION') or None
        if context not in {'v2','v3','v4'} or guard not in {None,'denial_v1'}:
            raise ValueError('invalid response context/guard')
        responder.context_version = context
        responder.fact_guard_version = guard
    return {'fact_extractor': extractor, 'response_mode': response_mode, 'responder': responder}
