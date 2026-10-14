"""Current development repeat caps, including per-fact presence clarification."""
from agent.planner import SLOT_ASK_LIMITS


def followup_limits(state):
    counts=state.get('problem_followup_counts')
    groups=list(counts.values()) if counts else [state.get('followup_counts',{})]
    for group in groups:
        for key,value in group.items():
            if type(value) is not int or value<0:return False
            # Aggregate count spans multiple facts, each with its own cap below.
            if key!='fact_confirmation' and value>SLOT_ASK_LIMITS.get(key,1):return False
    confirmations=state.get('fact_confirmation_counts',{})
    return all(type(n) is int and 0<=n<=2 for n in confirmations.values())
