"""Local-only recovery/circuit state for the opt-in prototype; old clients frozen."""
from threading import Lock
import time
from llm.clients import ModelClient,ModelResult
from llm.vllm_budget import BudgetedModelClient
from llm.vllm_budget import input_budget

def wire_schema(value):
    # vLLM 0.30 xgrammar rejects uniqueItems. Local JSON Schema still enforces it.
    if isinstance(value,dict):return {k:wire_schema(v) for k,v in value.items() if k!='uniqueItems'}
    if isinstance(value,list):return [wire_schema(v) for v in value]
    return value


_endpoints={}
_lock=Lock()


class LocalRecovery:
    def input_budget(self,messages,output_tokens):
        budget=input_budget(self.config,messages,output_tokens)
        self.last_input_budget=budget
        return budget
    def call(self,messages,**kwargs):
        cfg=self.config
        if cfg.provider!='vllm':return super().call(messages,**kwargs)
        with _lock:
            health=_endpoints.setdefault(cfg.api_url,{'failures':0,'open_until':0})
            if health['open_until']>time.monotonic():
                return ModelResult('',cfg.provider,cfg.model_id,{},0,'network_error',error_code='local_circuit_open')
        # No API fee reservation is uncertain for this local role; retry the next
        # new turn after recovery, without silently resetting paid-model limits.
        self.usage_unknown=False
        try:
            budget=self.input_budget(messages,cfg.max_tokens)
            if not budget['within_limit']:
                return ModelResult('',cfg.provider,cfg.model_id,{},0,'budget_exceeded',error_code='serving_context_limit')
        except Exception:
            with _lock:
                health['failures']+=1
                if health['failures']>=2:health['open_until']=time.monotonic()+30
            return ModelResult('',cfg.provider,cfg.model_id,{},0,'network_error',error_code='tokenizer_unavailable')
        if kwargs.get('json_schema') is not None:kwargs={**kwargs,'json_schema':wire_schema(kwargs['json_schema'])}
        result=super().call(messages,**kwargs)
        with _lock:
            if result.status in {'timeout','network_error'} or result.error_code in {'http_429','http_502','http_503','http_504'}:
                health['failures']+=1
                if health['failures']>=2:health['open_until']=time.monotonic()+30
            elif result.ok:health.update(failures=0,open_until=0)
        return result


class PrototypeClient(LocalRecovery,ModelClient):pass
class PrototypeResponder(LocalRecovery,BudgetedModelClient):pass
