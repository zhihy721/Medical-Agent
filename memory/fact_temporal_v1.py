"""Finite explicit onset interpretation; immutable utterance-time facts survive."""
import re
from llm.fact_normalization import SYMPTOM_ALIASES

VERSION = 'explicit_occurrence_v1'
FIELDS = {'symptoms','red_flags','sleep','sweating','appetite','thirst','cold_heat','stool_urine','emotion'}


def prior_denials_ended(inp, records, old_records, state):
    """Return prior IDs; never invent positives or correct model labels.

    A grounded current assertion with explicit now/onset wording ends an active
    older denial. Ambiguity, same-turn opposites, correction, quotations and
    conditional language deliberately abstain.
    """
    ended=[]
    for new in records:
        if (new['operation']!='assert' or new['field'] not in FIELDS
                or new['polarity']!='present' or new['subject']!='patient'
                or new['temporality']!='current'):
            continue
        evidence=new['evidence']
        if re.search(r'说错|记错|更正|改正|不确定|说不准|不知道|是否|是不是|如果|假设|要是|引用|[“”「」\"]',evidence):
            continue
        aliases={new['value']} | {a for a,v in SYMPTOM_ALIASES.items() if v==new['value']}
        names='|'.join(re.escape(a) for a in sorted(aliases,key=len,reverse=True))
        match=re.search(r'(?:现在|目前|这会儿|此刻)(?:确实有?|有|出现了?|开始有?|感到|感觉)?(?:'+names+r')',evidence)
        if not match:
            continue
        absolute=new['start']+match.start()
        context=re.split(r'[。！？；;\n]',inp['current_turn'][:absolute])[-1]
        if re.search(r'如果|假设|要是|引用|说错|记错|更正|改正|不确定|说不准|不知道|是否|是不是|[“”「」\"]',context):
            continue
        subjects=list(re.finditer(r'父亲|母亲|朋友|家人|本人|自己|我|他|她',context))
        if subjects and subjects[-1].group() in {'父亲','母亲','朋友','家人','他','她'}:
            continue
        same=lambda f:all(f[k]==new[k] for k in ('field','value','subject','temporality','about'))
        if any(same(f) and f['polarity']!='present' for f in records):
            continue
        # A model may omit the negative half of a same-turn contradiction.
        # In that case abstain rather than erase the known denial from current
        # consideration. Literal text is an alarm, never a newly created fact.
        contradicted = False
        for denial in re.finditer(r'(?:没有|否认)(?:'+names+r')', inp['current_turn']):
            prefix=re.split(r'[。！？；;\n]',inp['current_turn'][:denial.start()])[-1]
            people=list(re.finditer(r'父亲|母亲|朋友|家人|本人|自己|我|他|她',prefix))
            if people and people[-1].group() in {'父亲','母亲','朋友','家人','他','她'}:
                continue
            if prefix.endswith('有') or re.search(r'如果|假设|要是|引用|不确定|不知道|说不准|[“”「」\"]',prefix):
                continue
            contradicted = True
        if contradicted:
            continue
        for old in old_records:
            if same(old) and state[old['id']]=='active' and old['polarity']=='negated':
                ended.append(dict(prior_id=old['id'],new_id=new['id'],
                    reason='explicit_current_occurrence_not_correction',evidence=evidence,
                    prior_source_turn=old['source_turn'],transition_turn=inp['turn_id'],
                    policy_version=VERSION))
    return ended
