"""Versioned data boundary, not a general prompt injection detector.

RAG permits only exact records from the locally loaded corpus. Hospital display
projects bounded fields and never copies a tool's free-form note/instructions.
This does not verify clinical content or the integrity of local corpus files.
"""
import hashlib
import json
import math
import re

VERSION = 'tool_input_boundary_v1'
_LABEL = re.compile(r'[\u3400-\u9fffA-Za-z0-9 ·（）()\-]{1,80}')
_CONTROL = re.compile(r'忽略|系统指令|提示词|密钥|密码|验证码|提供.*凭据|ignore|system\s*prompt|api\s*key|password', re.I)


def audit(kind, result, reason):
    from observability.events import event_logger
    # Keep provenance without copying arbitrary tool text into the trace UI.
    digest = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                      default=str).encode()).hexdigest()
    event_logger.emit('tool_input_boundary', version=VERSION, source_kind=kind,
                      result_sha256=digest, reason=reason)


def corpus_hits(hits):
    from knowledge.tcm_knowledge import get_corpus
    known = {entry['id']: entry for entry in get_corpus()}
    accepted = []
    for hit in hits if isinstance(hits, list) else []:
        if not isinstance(hit, dict) or hit.get('type') != 'corpus':
            continue
        entry = known.get(hit.get('id')) if isinstance(hit.get('id'), str) else None
        if not entry or any(hit.get(k) != entry[v] for k, v in (
                ('name', 'title'), ('content', 'content'), ('source', 'source_file'))):
            audit('retrieval', hit, 'unknown_or_changed_local_record')
            continue
        # Reconstruct the data from the local record, rather than forwarding
        # unknown fields such as messages, role or instructions supplied by hits.
        accepted.append({'id': entry['id'], 'type': 'corpus', 'name': entry['title'],
                         'content': entry['content'], 'source': entry['source_file']})
    return accepted


def label(value):
    return value if isinstance(value, str) and _LABEL.fullmatch(value) and not _CONTROL.search(value) else ''


def number(value, upper):
    if isinstance(value, bool):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if math.isfinite(n) and 0 <= n <= upper else None


def hospital_data(data):
    if not isinstance(data, dict):
        audit('hospital', data, 'invalid_shape')
        return {}, []
    rows = []
    raw = data.get('hospitals', [])
    for item in raw[:5] if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not label(item.get('name')):
            audit('hospital', item, 'invalid_display_name')
            continue
        rows.append({'name': label(item['name']), 'district': label(item.get('district')),
                     'has_emergency': item.get('has_emergency') is True,
                     'distance_km': number(item.get('distance_km'), 1000)})
    # Never echo the free-form note. Source is an enum, not an instruction.
    source = data.get('source')
    source = source if isinstance(source, str) and source in {'amap_nearby', 'amap_nearby_fallback', 'demo', 'remote',
                                 'amap', 'demo_fallback', 'amap_fallback'} else 'unknown'
    return {'source': source, 'radius_km': number(data.get('radius_km'), 100) or 10}, rows
