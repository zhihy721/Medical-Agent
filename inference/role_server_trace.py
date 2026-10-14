"""Experiment-only vLLM ASGI/engine timestamps; no generation code is changed."""
import json
import os
from pathlib import Path
import threading
import time

_LOCK=threading.Lock()
_ACTIVE={}
_INSTALLED=False


def emit(value):
    path=Path(os.environ['ROLE_TRACE_FILE'])
    line=(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n').encode()
    with _LOCK:
        with path.open('ab') as stream:stream.write(line);stream.flush()


def engine_record(req_state,finish_reason):
    metrics=req_state.stats
    if metrics is None:return
    rid=req_state.external_req_id
    # X-Request-Id is used as the chatcmpl base; single choice appends '-0'.
    matches=[key for key in _ACTIVE if rid in {'chatcmpl-'+key,'chatcmpl-'+key+'-0'}]
    if len(matches)!=1:
        if 'rsm-' in rid:emit(dict(kind='unmatched_engine_request',engine_request_id=rid))
        return
    key=matches[0]
    emit(dict(kind='engine_finished',request_id=key,engine_request_id=rid,
        http_received_monotonic=_ACTIVE[key],frontend_finished_monotonic=time.monotonic(),
        engine_arrival_wall=metrics.arrival_time,frontend_first_token_latency_s=metrics.first_token_latency,
        queued_monotonic=metrics.queued_ts,scheduled_monotonic=metrics.scheduled_ts,
        first_token_monotonic=metrics.first_token_ts,last_token_monotonic=metrics.last_token_ts,
        prompt_tokens=req_state.prompt_len,output_tokens=metrics.num_generation_tokens,
        max_tokens=req_state.max_tokens_param,num_preemptions=metrics.num_preemptions,
        num_cached_tokens=req_state.num_cached_tokens,lora_name=req_state.lora_name,
        finish_reason=str(finish_reason)))


def install():
    global _INSTALLED
    if _INSTALLED:return
    from vllm.v1.engine.output_processor import OutputProcessor
    original=OutputProcessor._update_stats_from_finished
    def finished(self,req_state,finish_reason,iteration_stats):
        value=original(self,req_state,finish_reason,iteration_stats)
        engine_record(req_state,finish_reason)
        return value
    OutputProcessor._update_stats_from_finished=finished
    _INSTALLED=True


class TimingMiddleware:
    def __init__(self,app):
        self.app=app
        install()

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('path')!='/v1/chat/completions':
            return await self.app(scope,receive,send)
        rid=dict(scope.get('headers',[])).get(b'x-request-id',b'').decode('ascii',errors='replace')
        if not rid.startswith('rsm-'):return await self.app(scope,receive,send)
        begin=time.monotonic();_ACTIVE[rid]=begin
        emit(dict(kind='http_received',request_id=rid,monotonic=begin,wall=time.time()))
        status=None;ended=False
        async def observe(message):
            nonlocal status,ended
            if message['type']=='http.response.start':status=message['status']
            if message['type']=='http.response.body' and not message.get('more_body',False):
                ended=True
                emit(dict(kind='http_completed',request_id=rid,monotonic=time.monotonic(),status=status))
            await send(message)
        try:
            return await self.app(scope,receive,observe)
        finally:
            if not ended:emit(dict(kind='http_unfinished',request_id=rid,monotonic=time.monotonic(),status=status))
            _ACTIVE.pop(rid,None)
