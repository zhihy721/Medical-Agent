"""Single-answer Judge v2, with read-only evidence checks and explicit budget.

No automatic cloud calls at import. Pair preferences are produced separately
only after engineering checks and human adjudication permit comparison.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from jsonschema import ValidationError
from evaluation.hybrid_grader import validate_judgment

RUBRIC = Path(__file__).with_name('judge_rubric_v2.md')
KEYS = {'patient_history', 'policy_expectations', 'response', 'action', 'question_targets', 'tool_outputs'}


def fingerprint(context):
    # Identity is absent; all evaluation-relevant context contributes.
    return hashlib.sha256(json.dumps(context, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def messages(context):
    if not isinstance(context, dict) or set(context) != KEYS:
        raise ValueError('explicit blinded context required')
    if not isinstance(context['response'], str) or not isinstance(context['action'], str):
        raise ValueError('invalid response/action')
    for key in ('patient_history','policy_expectations','question_targets','tool_outputs'):
        if not isinstance(context[key], list) or any(not isinstance(v, str) for v in context[key]):
            raise ValueError('invalid context list')
    if not context['patient_history'] or not context['policy_expectations']:
        raise ValueError('patient evidence and reviewed policy required')
    return [{'role':'system','content':RUBRIC.read_text(encoding='utf8')+
        '\n只输出JSON：{"scores":{"relevance":0,"clarity":0,"groundedness":0},"issues":[],"abstain_reason":""}。'
        '\nissues各项包含type,severity,source,quote,reason。source是response或patient_history:N或tool_outputs:N；N从0开始。'},
        {'role':'user','content':json.dumps(context,ensure_ascii=False)}]


def check_quotes(value, context):
    value = validate_judgment(value)
    for issue in value['issues']:
        source = issue['source']
        if source == 'response':
            text = context['response']
        else:
            parts = source.split(':')
            if len(parts) != 2 or parts[0] not in {'patient_history','tool_outputs'} or not parts[1].isdigit():
                raise ValueError('judge_issue_source')
            index = int(parts[1])
            if index >= len(context[parts[0]]):
                raise ValueError('judge_issue_source')
            text = context[parts[0]][index]
        if issue['quote'] not in text:
            raise ValueError('judge_issue_quote_not_in_source')
    return value


def score_one(context, client, cache=None):
    """Cache is scoped to the exact rubric/config. Keep API errors in results."""
    from dataclasses import asdict
    from evaluation.hybrid_grader import JUDGE_SCHEMA
    request = messages(context)
    # Requiring a real RoleConfig prevents reuse across untracked model changes.
    cfg = client.config.public_summary()
    cache_key = fingerprint(dict(context=context,config=cfg,rubric=RUBRIC.read_text(encoding='utf8')))
    if cache is not None and cache_key in cache:
        return {**deepcopy(cache[cache_key]), 'cache_reused':True}
    # Existing cloud client accepts JSON-object requests, not vLLM schemas.
    options = {'json_schema':JUDGE_SCHEMA} if client.config.provider == 'vllm' else {}
    result = client.call(request,require_json=True,**options)
    record = dict(status=result.status,input_sha256=fingerprint(context),cache_key=cache_key,
                  cache_reused=False,model=asdict(result),judgment=None)
    if result.ok:
        try:
            record['judgment']=check_quotes(json.loads(result.content),context)
        except (ValueError,TypeError,KeyError,IndexError) as exc:
            record.update(status='invalid_score',error=str(exc))
        except ValidationError as exc:
            record.update(status='invalid_score',error='judge_schema:'+exc.message)
    if cache is not None and record['status']=='ok':
        cache[cache_key]=deepcopy(record)
    return record
