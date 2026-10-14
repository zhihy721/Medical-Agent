"""Finite onset interpretation with explicit policy versions and source guards."""
import re
from llm.fact_normalization import SYMPTOM_ALIASES
from memory import fact_temporal_v1 as legacy

VERSION = 'explicit_occurrence_v2'
SUPPORTED_VERSIONS = {legacy.VERSION, VERSION}
FIELDS = legacy.FIELDS
BOUNDARY = r'[。！？；;\n]'
BLOCKED = r'说错|记错|更正|改正|不确定|说不准|不知道|是否|是不是|如果|假设|要是|假如|倘若|可能|也许|或许|似乎|好像|引用|[“”「」\"]'
PEOPLE = r'父亲|母亲|朋友|家人|本人|自己|我|他|她'
OTHERS = {'父亲', '母亲', '朋友', '家人', '他', '她'}


def _other_subject(prefix):
    people = list(re.finditer(PEOPLE, prefix))
    return bool(people and people[-1].group() in OTHERS)


def _source_allows_onset(text, absolute, end):
    """Check source on both sides even if model evidence was shortened.

    Ambiguous scope conservatively abstains; no new facts or replacement IDs
    are inferred here.
    """
    prefix = re.split(BOUNDARY, text[:absolute])[-1]
    suffix = re.split(BOUNDARY, text[end:])[0]
    sentence = prefix + text[absolute:end] + suffix
    if re.search(BLOCKED, sentence) or re.search(r'[?？]', text[end:end+len(suffix)+1]):
        return False
    if re.match(r'(?:的|了)?(?:吗|么|吧)', suffix):
        return False
    if _other_subject(prefix) or re.search(r'说的是(?:我)?(?:父亲|母亲|朋友|家人|他|她)', suffix):
        return False
    if re.search(r'(?:不是|并非|没有|并无|没|不|无)\s*$', prefix):
        return False
    return True


def prior_denials_ended(inp, records, old_records, state, *, policy_version=VERSION, identity_version=None, lexical_aliases=None):
    """Append temporal events only for already extracted grounded assertions."""
    if policy_version == legacy.VERSION:
        return legacy.prior_denials_ended(inp, records, old_records, state)
    if policy_version != VERSION:
        raise ValueError('unsupported temporal policy version')
    ended = []
    old_records = list(old_records)
    text = inp['current_turn']
    for new in records:
        if (new['operation']!='assert' or new['field'] not in FIELDS
                or new['polarity']!='present' or new['subject']!='patient'
                or new['temporality']!='current'):
            continue
        evidence=new['evidence']
        if re.search(BLOCKED, evidence):
            continue
        aliases={new['value']} | {a for a,v in SYMPTOM_ALIASES.items() if v==new['value']}
        aliases |= {a for a,v in (lexical_aliases or {}).items() if v==new['value']}
        names = '(?:' + '|'.join(re.escape(a) for a in sorted(aliases,key=len,reverse=True)) + ')'
        now = r'(?:现在|目前|这会儿|此刻)(?:确实有?|有|新出现了?|出现了?|开始(?:有|出现)?|感到|感觉)?' + names
        before = r'(?:刚刚|刚才|刚)(?:才)?(?:开始(?:有|出现)?|出现了?)' + names
        after = names + r'(?:是)?(?:才)?(?:刚刚|刚才|刚|现在|目前|这会儿|此刻)(?:才|新)?(?:开始|出现)(?:的|了)?'
        match = re.search(now + '|' + before + '|' + after, evidence)
        if not match:
            continue
        absolute=new['start']+match.start()
        end = new['start'] + match.end()
        if not _source_allows_onset(text, absolute, end):
            continue
        from memory.fact_identity import same_concept
        same=lambda f:(same_concept(f,new,version=identity_version)
                       and all(f[k]==new[k] for k in ('subject','temporality','about')))
        if any(same(f) and f['polarity']!='present' for f in records):
            continue
        # A model may omit the negative half of a same-turn contradiction.
        # In that case abstain rather than erase the known denial from current
        # consideration. Literal text is an alarm, never a newly created fact.
        contradicted = False
        for denial in re.finditer(r'(?:没有|否认|并无|不存在|没)(?:出现|有)?' + names, text):
            prefix=re.split(BOUNDARY, text[:denial.start()])[-1]
            if (_other_subject(prefix) or prefix.endswith('有')
                    or re.search(r'如果|假设|要是|假如|倘若|引用|不确定|不知道|说不准|[“”「」\"]',prefix)):
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
