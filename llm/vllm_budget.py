"""Opt-in input measurement using the serving endpoint's own chat tokenizer.

No text is shortened. Errors propagate to the renderer's template fallback.
This wrapper is specific to the text-only non-thinking experiment, not a cloud
tokenizer estimate or a replacement for any locked extraction evaluator.
"""
import hashlib
import json
from urllib.parse import urlsplit

import requests
from llm.clients import ModelClient


class BudgetedModelClient(ModelClient):
    """Explicit opt-in for the app; ordinary ModelClient stays unchanged."""
    def input_budget(self, messages, output_tokens):
        return input_budget(self.config, messages, output_tokens)


def input_budget(config, messages, output_tokens, context_limit=4096):
    if config.provider!='vllm' or config.enable_thinking is not False:
        raise ValueError('budget measurement requires non-thinking vLLM')
    url=urlsplit(config.api_url)
    if url.hostname not in {'127.0.0.1','localhost','::1'}:
        raise ValueError('budget measurement restricted to local SSH tunnel')
    response=requests.post(f'{url.scheme}://{url.netloc}/tokenize',json={
        'model':config.model_id,'messages':messages,'add_generation_prompt':True,
        'chat_template_kwargs':{'enable_thinking':False}},timeout=min(config.timeout_s,30))
    response.raise_for_status()
    tokens=response.json().get('tokens')
    if not isinstance(tokens,list) or not all(type(t) is int for t in tokens):
        raise ValueError('tokenizer returned invalid token list')
    count=len(tokens)
    return {'method':'serving_vllm_chat_tokenizer_v1','prompt_tokens':count,
        'output_tokens':output_tokens,'context_limit':context_limit,
        'within_limit':count+output_tokens<=context_limit,
        'input_sha256':hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
