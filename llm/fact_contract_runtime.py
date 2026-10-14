"""Versioned online fact contract; locked training V1 remains byte-identical."""
from copy import deepcopy
import json
from jsonschema import Draft202012Validator
from llm import fact_contract as legacy
from llm.fact_contract import (VERSION, FACT_FIELDS, ATTRIBUTES, POLARITIES, TIMES,
    SUBJECTS, KNOWN_KEYS, DELTA_SCHEMA, FACT_SCHEMA, RED_FLAG_VALUES, PROMPT,
    derived_fields)
V2 = 'semantic_facts_v2'
OBSERVATION_FIELDS = {'sleep', 'sweating', 'appetite', 'thirst', 'cold_heat', 'stool_urine', 'emotion'}

def build_input(current_turn, turn_id=1, known_facts=None, last_question='', current_focus=None, *, contract_version=VERSION):
    if contract_version == VERSION:
        return legacy.build_input(current_turn, turn_id, known_facts, last_question, current_focus)
    if contract_version not in {VERSION, V2}:
        raise ValueError('unsupported fact contract')
    if not isinstance(current_turn, str) or not current_turn.strip() or len(current_turn) > 4096:
        raise ValueError('current_turn requires 1..4096 characters; no silent truncation')
    if type(turn_id) is not int or turn_id < 1:
        raise ValueError('invalid turn_id')
    known = deepcopy(known_facts) if known_facts is not None else []
    if not isinstance(known, list) or len(known) > 32:
        raise ValueError('known_facts exceeds 32; context compaction required')
    seen = set()
    for fact in known:
        if not isinstance(fact, dict) or set(fact) != KNOWN_KEYS:
            raise ValueError('known fact requires versioned provenance keys')
        if not isinstance(fact['id'], str) or not fact['id'] or fact['id'] in seen:
            raise ValueError('known fact IDs must be unique nonempty strings')
        seen.add(fact['id'])
        if (fact['field'] not in FACT_FIELDS or fact['polarity'] not in POLARITIES
                or fact['subject'] not in SUBJECTS or fact['temporality'] not in TIMES
                or fact['status'] not in {'active', 'disputed'}
                or not isinstance(fact['value'], str) or not 0 < len(fact['value']) <= 120):
            raise ValueError('invalid known fact semantics')
        if fact['source'] not in {'current_dialogue', 'manual_profile', 'profile_prefill'}:
            raise ValueError('unknown fact source')
        source_turn = fact['source_turn']
        if type(source_turn) is not int or not 0 <= source_turn < turn_id:
            raise ValueError('known fact source must precede current turn')
        if (fact['source'] == 'current_dialogue') != (source_turn > 0):
            raise ValueError('dialogue uses positive turns; profile uses turn zero')
    for fact in known:
        ref = fact['about']
        if ref is not None:
            if ref not in seen or fact['field'] not in ATTRIBUTES:
                raise ValueError('invalid known attribute reference')
            _check_about(fact, next(f for f in known if f['id'] == ref), contract_version)
    if not isinstance(last_question, str) or len(last_question) > 1024:
        raise ValueError('last_question exceeds cap')
    if current_focus is not None:
        focus = next((f for f in known if f['id'] == current_focus), None)
        if not focus or focus['field'] not in {'symptoms', 'red_flags'}:
            raise ValueError('current_focus must reference a known problem')
    if len(json.dumps(known, ensure_ascii=False)) > 12000:
        raise ValueError('known_facts exceeds character cap; tokenize before training')
    return {'contract_version': contract_version, 'current_turn': current_turn, 'turn_id': turn_id,
            'known_facts': known, 'last_question': last_question, 'current_focus': current_focus}

def validate_input(inp):
    if not isinstance(inp, dict) or set(inp) != {'contract_version', 'current_turn', 'turn_id',
                                               'known_facts', 'last_question', 'current_focus'}:
        raise ValueError('unexpected input keys')
    if inp['contract_version'] not in {VERSION, V2}:
        raise ValueError('contract version mismatch')
    build_input(inp['current_turn'], inp['turn_id'], inp['known_facts'], inp['last_question'], inp['current_focus'], contract_version=inp['contract_version'])

def _check_about(fact, problem, version=VERSION):
    allowed = {'symptoms', 'red_flags'}
    if version == V2 and fact['field'] == 'duration':
        allowed |= OBSERVATION_FIELDS
    if (problem['field'] not in allowed or fact['subject'] != problem['subject']
            or problem['polarity'] != 'present' or fact['temporality'] != problem['temporality']):
        raise ValueError('attribute must reference a positive problem of same subject/time')

def position_delta(delta, inp):
    """Return new grounded records; never edit the raw model output."""
    if inp.get('contract_version') == VERSION:
        return legacy.position_delta(delta, inp)
    validate_input(inp)
    errors = list(Draft202012Validator(DELTA_SCHEMA).iter_errors(delta))
    if errors:
        raise ValueError('fact_schema: ' + errors[0].message)
    facts = delta['facts']
    if inp['contract_version'] == V2:
        for f in facts:
            if f['value'].strip().lower() in {'null', 'none', 'n/a', '不确定', '未知'}:
                raise ValueError('placeholder is not a semantic fact value; use uncertainty on the fact')
    local = {f['id']: f for f in facts}
    if len(local) != len(facts):
        raise ValueError('duplicate local fact ID')
    known = {f['id']: f for f in inp['known_facts']}
    if set(local) & set(known):
        raise ValueError('local ID collides with known ID')
    records, seen, replacements = [], set(), set()
    text = inp['current_turn']
    for fact in facts:
        if fact['field'] == 'red_flags' and fact['value'] not in RED_FLAG_VALUES:
            raise ValueError('red flag outside versioned engineering vocabulary')
        evidence = fact['evidence']
        start = text.find(evidence)
        if start < 0 or text.find(evidence, start + 1) >= 0:
            raise ValueError('evidence must occur exactly once; expand repeated evidence')
        about = fact['about']
        if about is not None:
            problem = local.get(about) or known.get(about)
            if not problem or fact['field'] not in ATTRIBUTES:
                raise ValueError('unknown or unsupported about reference')
            _check_about(fact, problem, inp['contract_version'])
        target = fact['replaces']
        if fact['operation'] == 'assert':
            if target is not None:
                raise ValueError('assert cannot replace an old fact')
        else:
            old = known.get(target)
            if not old or target in replacements:
                raise ValueError('replacement requires a unique known target')
            replacements.add(target)
            same_binding = old['about'] == about
            if old['about'] is None and fact['field'] in ATTRIBUTES and about is not None:
                same_binding = True  # Explicitly resolving a previously unbound attribute.
            if not same_binding and old['about'] in known and about in local:
                previous, repeated = known[old['about']], local[about]
                same_binding = all(previous[k] == repeated[k] for k in
                                   ('field', 'value', 'polarity', 'subject', 'temporality'))
            if (old['field'], old['subject']) != (fact['field'], fact['subject']) or not same_binding:
                raise ValueError('correction must preserve field, subject and problem binding')
            if fact['operation'] == 'resolve':
                resolvable_fields = {'symptoms', 'red_flags'} | (OBSERVATION_FIELDS if inp['contract_version'] == V2 else set())
                if (fact['field'] not in resolvable_fields or fact['temporality'] != 'resolved'
                        or fact['polarity'] != 'present' or old['polarity'] != 'present'
                        or old['temporality'] != 'current' or fact['value'] != old['value']):
                    raise ValueError('resolve requires a previously positive current problem')
            elif fact['temporality'] != old['temporality']:
                raise ValueError('correction cannot silently move a fact between time scopes')
        key = tuple(fact[k] for k in ('field', 'value', 'polarity', 'subject', 'temporality', 'about'))
        if key in seen:
            raise ValueError('duplicate semantic fact')
        seen.add(key)
        record = deepcopy(fact)
        record.update(id=f"t{inp['turn_id']}:{fact['id']}", start=start, end=start + len(evidence),
                      source='current_dialogue', source_turn=inp['turn_id'])
        if about in local:
            record['about'] = f"t{inp['turn_id']}:{about}"
        records.append(record)
    # Deterministic field copies are not additional model targets.
    for fact in facts:
        copies = derived_fields(fact)
        if any(other is not fact and (other['field'], other['value']) in copies
               and all(other[k] == fact[k] for k in ('polarity', 'subject', 'temporality', 'evidence'))
               for other in facts):
            raise ValueError('derived field repeated in model target')
    return records

def decoding_schema(inp):
    """Ground structural references in the actual input, never in expected labels.

    Syntax-only alternatives prevent self-linked non-attributes and unknown old
    targets. They cannot decide which fact is true or which old record to correct.
    """
    if inp.get('contract_version') == VERSION:
        return legacy.decoding_schema(inp)
    validate_input(inp)
    known_ids = [f['id'] for f in inp['known_facts']]
    problem_ids = [f['id'] for f in inp['known_facts']
                   if f['field'] in {'symptoms', 'red_flags'} and f['polarity'] == 'present']
    relation_fields = {'symptoms', 'red_flags'} | (OBSERVATION_FIELDS if inp['contract_version'] == V2 else set())
    duration_ids = [f['id'] for f in inp['known_facts'] if f['field'] in relation_fields and f['polarity'] == 'present']
    refs = [None, *duration_ids, *[f'f{i}' for i in range(1, 65)]]
    variants = []
    groups = [sorted(set(FACT_FIELDS) - ATTRIBUTES - {'red_flags'}), ['duration'],
              sorted(ATTRIBUTES - {'duration'}), ['red_flags']]
    for fields in groups:
        for operation in (['assert', 'correct'] if known_ids else ['assert']):
            item = deepcopy(FACT_SCHEMA)
            props = item['properties']
            props['field'] = {'enum': fields}
            props['operation'] = {'const': operation}
            attribute_group = set(fields).issubset(ATTRIBUTES)
            binding_refs = refs if fields == ['duration'] else [None, *problem_ids, *[f'f{i}' for i in range(1,65)]]
            props['about'] = {'enum': binding_refs if attribute_group else [None]}
            # Duration semantics belong to the prompt and grounded checks. A
            # units-only grammar excludes legitimate vague patient wording
            # such as "很久" and cannot prevent narrative strings with "天".
            props['replaces'] = {'enum': known_ids if operation == 'correct' else [None]}
            if fields == ['red_flags']:
                props['value'] = {'enum': sorted(RED_FLAG_VALUES)}
            variants.append(item)
    resolvable = [f['id'] for f in inp['known_facts'] if f['id'] in duration_ids and f['temporality'] == 'current']
    if resolvable:
        for field in sorted(relation_fields):
            item = deepcopy(FACT_SCHEMA)
            props = item['properties']
            props.update(field={'const': field}, operation={'const': 'resolve'}, about={'type': 'null'},
                         replaces={'enum': resolvable}, temporality={'const': 'resolved'}, polarity={'const': 'present'})
            if field == 'red_flags':
                props['value'] = {'enum': sorted(RED_FLAG_VALUES)}
            variants.append(item)
    schema = deepcopy(DELTA_SCHEMA)
    schema['properties']['facts']['items'] = {'anyOf': variants}
    return schema


def messages(inp, target=None):
    validate_input(inp)
    if inp['contract_version'] == VERSION:
        return legacy.messages(inp, target)
    from llm.semantic_prompt import V3
    result = [{'role':'system','content':V3},
              {'role':'user','content':json.dumps(inp,ensure_ascii=False,separators=(',',':'))}]
    if target is not None:
        position_delta(target,inp)
        result.append({'role':'assistant','content':json.dumps(target,ensure_ascii=False,separators=(',',':'))})
    return result
