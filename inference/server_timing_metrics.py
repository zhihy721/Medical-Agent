"""Server-only timing joins; missing traces never fall back to client/SSE clocks."""
from inference.benchmark_vllm import percentile

def metric_values(text):
    result={}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            try:key,value=line.rsplit(None,1);result[key]=float(value)
            except ValueError:continue
    return result

def one(events, kind):
    selected = [e for e in events if e['kind'] == kind]
    if len(selected) > 1:
        raise ValueError('duplicate server event: ' + kind)
    return selected[0] if selected else None

def measure(result, events):
    """All latency timestamps are from one Linux host's monotonic clock."""
    received = one(events, 'http_received')
    engine = one(events, 'engine_finished')
    complete = one(events, 'http_completed')
    errors = []
    if not received or not engine or not complete:
        return dict(status='missing', errors=['missing_' + k for k, v in
            [('http_received', received), ('engine_finished', engine), ('http_completed', complete)] if not v])
    t = [received['monotonic'], engine['queued_monotonic'], engine['scheduled_monotonic'],
         engine['first_token_monotonic'], engine['last_token_monotonic'],
         engine['frontend_finished_monotonic'], complete['monotonic']]
    if any(v is None or not isinstance(v, (int, float)) for v in t):
        return dict(status='invalid', errors=['timestamp_missing_or_not_numeric'])
    if any(a > b + 1e-6 for a, b in zip(t, t[1:])):
        errors.append('server_timestamps_out_of_order')
    if abs(engine['http_received_monotonic'] - received['monotonic']) > 1e-6:
        errors.append('server_receive_timestamp_mismatch')
    if complete['status'] != result.get('status_code'):
        errors.append('http_status_mismatch')
    n = engine['output_tokens']
    if not isinstance(n, int) or n < 0:
        return dict(status='invalid', errors=errors + ['invalid_output_token_count'])
    values = dict(server_ttft_ms=(t[3]-t[0])*1000, queue_ms=(t[2]-t[1])*1000,
                  prefill_ms=(t[3]-t[2])*1000,
                  average_tpot_ms=(t[4]-t[3])*1000/(n-1) if n > 1 else None,
                  server_last_token_ms=(t[4]-t[0])*1000,
                  server_http_complete_ms=(t[6]-t[0])*1000,
                  api_prepare_ms=(t[1]-t[0])*1000,
                  request_tokens_per_s=n/(t[4]-t[2]) if t[4] > t[2] else None)
    response = result.get('response', {})
    native = response.get('metrics') or {}
    checks = dict(time_to_first_token_ms=values['prefill_ms'], queue_time_ms=values['queue_ms'],
                  generation_time_ms=(t[4]-t[3])*1000, mean_itl_ms=values['average_tpot_ms'],
                  tokens_per_second=values['request_tokens_per_s'])
    deltas = {}
    for key, expected in checks.items():
        actual = native.get(key)
        if expected is None:
            if actual is not None:
                errors.append('native_null_mismatch:' + key)
        elif actual is None:
            errors.append('native_missing:' + key)
        else:
            deltas[key] = actual-expected
            if abs(actual-expected) > 1e-4:
                errors.append('native_metric_mismatch:' + key)
    usage = response.get('usage') or {}
    if usage.get('completion_tokens') != n or usage.get('prompt_tokens') != engine['prompt_tokens']:
        errors.append('usage_count_mismatch')
    return dict(status='valid' if not errors else 'invalid', errors=errors, **values,
                prompt_tokens=engine['prompt_tokens'], output_tokens=n,
                num_cached_tokens=engine['num_cached_tokens'], num_preemptions=engine['num_preemptions'],
                lora_name=engine['lora_name'], engine_finish_reason=engine['finish_reason'],
                native_deltas=deltas, native_speculative=native.get('speculative_decoding'),
                begin_monotonic=t[0], end_monotonic=t[6], receive_wall=received['wall'])

def counters(before, after):
    a=metric_values(before); b=metric_values(after)
    def delta(prefix):
        return sum(v-a.get(k,0) for k,v in b.items() if k.startswith(prefix+'{'))
    drafted=delta('vllm:spec_decode_num_draft_tokens_total')
    accepted=delta('vllm:spec_decode_num_accepted_tokens_total')
    return dict(draft_tokens=drafted, accepted_tokens=accepted,
                acceptance=accepted/drafted if drafted else None,
                draft_rounds=delta('vllm:spec_decode_num_drafts_total'),
                prefix_cache_hits=delta('vllm:prefix_cache_hits_total'))
