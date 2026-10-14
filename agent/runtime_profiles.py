"""Explicit, shared delivery configuration; historical routes stay available."""
from pathlib import Path

STRATEGIES = {'rule_rule': ('rule', 'rule'), 'llm_rule': ('llm', 'rule'),
              'llm_llm': ('llm', 'llm')}
DESKTOP_STRATEGIES = {'g6-rule': 'rule_rule', 'g6-plan': 'llm_rule', 'g6-review': 'llm_llm'}


def delivery_settings(root, strategy='rule_rule'):
    if strategy not in STRATEGIES:
        raise ValueError('unknown delivery strategy')
    from llm.fact_contract_runtime import V2
    from memory.runtime_fact_projection import VERSION
    planner, reviewer = STRATEGIES[strategy]
    return {
        'AGENT_RUNTIME': 'langgraph', 'AGENT_PLANNER_MODE': planner,
        'AGENT_REVIEWER_MODE': reviewer, 'AGENT_POLICY_PROMPT_VERSION': 'v3',
        'AGENT_EXTRACTOR_MODE': 'semantic', 'AGENT_INTENT_MODE': 'model',
        'AGENT_INTENT_PROMPT_VERSION': 'v4', 'AGENT_INTENT_GUARD_VERSION': 'v2',
        'AGENT_INTENT_CHOICE_PROMPT_VERSION': 'v5', 'AGENT_INTENT_STRUCTURED_MODE': 'whole_turn_v1',
        'AGENT_EXTRACTOR_BOUNDARY_VERSION': 'v1', 'AGENT_EXTRACTOR_PROMPT_VERSION': 'g6_v7',
        'AGENT_EXTRACTOR_RECOVERY_MODE': 'checked_v16', 'AGENT_FACT_CONTRACT_VERSION': V2,
        'AGENT_REPRESENTATION_MODE': 'finite_v5', 'AGENT_FACT_PROJECTION_VERSION': VERSION,
        'AGENT_RESPONDER_MODE': 'model', 'AGENT_RESPONDER_BUDGET_MODE': 'serving_tokenizer',
        'AGENT_RESPONDER_CONTEXT_VERSION': 'v4', 'AGENT_RESPONDER_FACT_GUARD_VERSION': 'denial_v1',
        'AGENT_MODEL_ROLES_CONFIG': str(Path(root).resolve() / 'llm/semantic_roles.delivery_g6.yaml'),
    }


def full888_settings(root, strategy='rule_rule'):
    result=delivery_settings(root,strategy)
    result['AGENT_MODEL_ROLES_CONFIG']=str(Path(root).resolve()/'llm/semantic_roles.delivery_full888.yaml')
    result['AGENT_DELIVERY_GROUNDING_GUARD']='delivery_grounding_guard_v1'
    return result


def full888_safe_settings(root,strategy='rule_rule'):
    result=full888_settings(root,strategy)
    result['AGENT_DELIVERY_INPUT_SAFETY']='literal_breathing_safety_v2'
    result['AGENT_DELIVERY_TURN_REQUESTS']='delivery_turn_requests_v1'
    return result


def qwen_prototype_settings(root):
    result=full888_safe_settings(root,'llm_llm')
    result.update(AGENT_INTERACTION_MODE='joint_v1',
                  AGENT_MODEL_ROLES_CONFIG=str(Path(root).resolve()/'llm/semantic_roles.qwen_prototype.yaml'))
    return result


def qwen_grounded_settings(root):
    result=qwen_prototype_settings(root)
    result.update(AGENT_QWEN_PLAN_CONTRACT='basis_v1',AGENT_QWEN_REVIEW_REASON='visible')
    return result
