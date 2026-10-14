"""Measure an OpenAI-compatible streaming vLLM endpoint on a fixed workload.

Only a successful response with API token usage enters token throughput metrics.
This is an infrastructure benchmark, not a model-quality evaluation.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import requests


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower, upper = math.floor(rank), math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def consume_stream(lines, started, now=time.perf_counter):
    """Return timings from SSE. Empty role/reasoning chunks do not start TTFT."""
    first = last = None
    usage = None
    done = False
    content_chunks = 0
    finish_reason = None
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        if not raw.startswith('data: '):
            continue
        payload = raw[6:].strip()
        if payload == '[DONE]':
            done = True
            break
        item = json.loads(payload)
        if item.get('error'):
            raise ValueError('stream returned an error: '+str(item['error'])[:200])
        usage = item.get('usage') or usage
        for choice in item.get('choices', []):
            finish_reason = choice.get('finish_reason') or finish_reason
            content = choice.get('delta', {}).get('content')
            if content:
                instant = now()
                if first is None:
                    first = instant
                last = instant
                content_chunks += 1
    ended = now()
    if not done:
        raise ValueError('stream ended without [DONE]')
    if first is None:
        raise ValueError('stream contained no visible content tokens')
    output_tokens = usage.get('completion_tokens') if isinstance(usage, dict) else None
    input_tokens = usage.get('prompt_tokens') if isinstance(usage, dict) else None
    if type(output_tokens) is not int or output_tokens < 1:
        raise ValueError('stream omitted positive completion_tokens usage')
    if type(input_tokens) is not int or input_tokens < 0:
        raise ValueError('stream omitted prompt_tokens usage')
    return dict(ttft_seconds=first-started, e2e_seconds=ended-started,
                tpot_seconds=(ended-first)/(output_tokens-1) if output_tokens > 1 else None,
                prompt_tokens=input_tokens, completion_tokens=output_tokens,
                content_chunks=content_chunks, finish_reason=finish_reason)


def request_one(endpoint, model, row, max_tokens, timeout, api_key, now=time.perf_counter):
    started = now()
    payload = dict(model=model, messages=row['messages'], stream=True,
                   stream_options={'include_usage': True}, max_tokens=max_tokens,
                   temperature=0, chat_template_kwargs={'enable_thinking': False})
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer '+api_key
    try:
        with requests.post(endpoint, json=payload, headers=headers,
                           stream=True, timeout=(5, timeout)) as response:
            response.raise_for_status()
            result = consume_stream(response.iter_lines(decode_unicode=True), started, now)
        return dict(sample_id=row['sample_id'], status='ok', **result)
    except (requests.RequestException, ValueError, json.JSONDecodeError, UnicodeError) as exc:
        return dict(sample_id=row['sample_id'], status='failed',
                    error=type(exc).__name__+': '+str(exc)[:300],
                    e2e_seconds=now()-started)


def summarize(rows, wall_seconds):
    passed = [r for r in rows if r['status']=='ok']
    ttft = [r['ttft_seconds'] for r in passed]
    e2e = [r['e2e_seconds'] for r in passed]
    tpot = [r['tpot_seconds'] for r in passed if r['tpot_seconds'] is not None]
    tokens = sum(r['completion_tokens'] for r in passed)
    finishes = Counter(r.get('finish_reason') or 'unknown' for r in passed)
    return dict(requests=len(rows), succeeded=len(passed), failed=len(rows)-len(passed),
                finish_reasons=dict(finishes),
                ttft_p50_seconds=percentile(ttft,.5), ttft_p95_seconds=percentile(ttft,.95),
                e2e_p50_seconds=percentile(e2e,.5), e2e_p95_seconds=percentile(e2e,.95),
                tpot_p50_seconds=percentile(tpot,.5), tpot_p95_seconds=percentile(tpot,.95),
                tpot_eligible=len(tpot), wall_seconds=wall_seconds,
                requests_per_second=len(passed)/wall_seconds if wall_seconds > 0 else None,
                aggregate_output_tokens_per_second=tokens/wall_seconds if wall_seconds > 0 else None,
                output_tokens=tokens)


def read_workload(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('workload is empty or has duplicate sample IDs')
    for row in rows:
        if not isinstance(row.get('messages'), list) or not row['messages']:
            raise ValueError('every workload row needs nonempty messages')
        if any(message.get('role') not in {'system','user','assistant'} or
               not isinstance(message.get('content'),str) for message in row['messages']):
            raise ValueError('invalid workload message')
    return rows


def run(endpoint, model, workload, out, *, concurrencies=(1,2,4), rounds=3,
        warmups=1, max_tokens=128, timeout=120, api_key=''):
    rows = read_workload(workload)
    if not endpoint.startswith(('http://127.0.0.1:', 'http://localhost:')):
        raise ValueError('endpoint must be the local SSH tunnel or loopback vLLM service')
    if not model or rounds < 1 or warmups < 0 or max_tokens < 1 or timeout < 1:
        raise ValueError('invalid benchmark parameters')
    if not concurrencies or any(type(c) is not int or c < 1 for c in concurrencies):
        raise ValueError('invalid concurrency')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(scope='streaming_infrastructure_benchmark_not_quality',
                status='running', started_at_utc=utc_now(), model=model, endpoint=endpoint,
                workload_sha256=sha(workload),
                script_sha256=sha(__file__), workload_size=len(rows), rounds=rounds,
                warmups=warmups, max_tokens=max_tokens, timeout_seconds=timeout,
                concurrencies=list(concurrencies), temperature=0, enable_thinking=False,
                token_count_source='API stream_options.include_usage',
                timing_source='time.perf_counter', resource_measurement='not_collected',
                results=[])
    path = out/'report.json'
    path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        for _ in range(warmups):
            warm = request_one(endpoint,model,rows[0],max_tokens,timeout,api_key)
            if warm['status'] != 'ok':
                raise ValueError('warmup failed: '+warm['error'])
        for concurrency in concurrencies:
            jobs = [row for _ in range(rounds) for row in rows]
            batch_started_at_utc = utc_now()
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(request_one,endpoint,model,row,max_tokens,timeout,api_key)
                           for row in jobs]
                raw = [future.result() for future in futures]
            elapsed = time.perf_counter()-start
            item = dict(concurrency=concurrency, started_at_utc=batch_started_at_utc,
                        finished_at_utc=utc_now(), summary=summarize(raw,elapsed),raw=raw)
            meta['results'].append(item)
            path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        meta['status']='completed' if all(x['summary']['failed']==0 for x in meta['results']) else 'completed_with_failures'
    except BaseException as exc:
        meta['status']='failed';meta['error']=type(exc).__name__+': '+str(exc)[:300]
        raise
    finally:
        meta['finished_at_utc'] = utc_now()
        path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',default='http://127.0.0.1:8000/v1/chat/completions')
    parser.add_argument('--model',required=True)
    parser.add_argument('--workload',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--concurrencies',type=int,nargs='+',default=[1,2,4])
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--max-tokens',type=int,default=128)
    parser.add_argument('--timeout',type=int,default=120)
    args=parser.parse_args()
    import os
    report=run(args.endpoint,args.model,args.workload,args.out,
               concurrencies=args.concurrencies,rounds=args.rounds,warmups=args.warmups,
               max_tokens=args.max_tokens,timeout=args.timeout,
               api_key=os.getenv('VLLM_API_KEY',''))
    print(json.dumps({'status':report['status'],'results':[
        {'concurrency':r['concurrency'],**r['summary']} for r in report['results']]},ensure_ascii=False))


if __name__=='__main__':
    main()
