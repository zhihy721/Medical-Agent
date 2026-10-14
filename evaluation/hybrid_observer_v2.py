"""Development observer adapter; frozen v1/Judge source stays reproducible."""
from evaluation.hybrid_grader import observe_trajectory_row as observe_v1


def observe_trajectory_row(row):
    observed=observe_v1(row)
    confirmation=(row.get('plan') or {}).get('fact_confirmation') or {}
    if 'candidate_concept' in confirmation:
        observed['confirmation_value']=confirmation['candidate_concept']
    return observed
