"""Technical text-LoRA smoke, not an SFT quality experiment. Run in isolated WSL env."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import time


def language_targets(names):
    pattern=r'.*language_model\.layers\.\d+\.(self_attn\.(q_proj|v_proj)|linear_attn\.in_proj_qkv)'
    selected=[name for name in names if re.fullmatch(pattern,name)]
    if not selected or any('vision' in n or 'visual' in n for n in selected):
        raise ValueError('No verified text projection modules found')
    return selected


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.steps<=10:parser.error('technical smoke only: 1..10 steps')
    args.out.mkdir(parents=True,exist_ok=False)
    report={'status':'starting','scope':'technical smoke, authored draft data; no quality claim','steps_requested':args.steps,
            'data_sha256':hash_file(args.data),'script_sha256':hash_file(__file__),'training_loss':[]}
    def save():
        (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    started=time.time()
    try:
        import resource
        import torch
        import transformers,peft,accelerate
        from transformers import AutoModelForImageTextToText,AutoTokenizer
        from peft import LoraConfig,get_peft_model,PeftModel,TaskType
        torch.manual_seed(42)
        report['versions']={m.__name__:m.__version__ for m in (torch,transformers,peft,accelerate)}
        report['base_file_sha256']={p.name:hash_file(p) for p in args.model.iterdir() if p.suffix=='.safetensors' or p.name in {'config.json','model.safetensors.index.json','tokenizer_config.json'}}
        tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
        records=[json.loads(line) for line in args.data.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        tokenized=[]
        for r in records:
            msgs=r['messages']
            prefix=tok.apply_chat_template(msgs[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            whole=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False,enable_thinking=False,return_dict=False)
            if whole[:len(prefix)]!=prefix or len(whole)>768 or len(whole)<=len(prefix):
                raise ValueError('assistant boundary or length failure')
            tokenized.append({'input_ids':whole,'labels':[-100]*len(prefix)+whole[len(prefix):]})
        def base():
            return AutoModelForImageTextToText.from_pretrained(args.model,torch_dtype=torch.bfloat16,
                device_map={'':'cuda:0'},local_files_only=True,trust_remote_code=False,attn_implementation='sdpa')
        model=base()
        targets=language_targets([name for name,_ in model.named_modules()])
        report['lora_targets']=targets
        cfg=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,target_modules=targets,bias='none',task_type=TaskType.CAUSAL_LM)
        model=get_peft_model(model,cfg)
        model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in n or 'language_model' not in n for n,p in trainable):
            raise ValueError('Unexpected trainable parameters')
        report['trainable_parameters']=sum(p.numel() for _,p in trainable)
        report['total_parameters']=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.AdamW([p for _,p in trainable],lr=1e-4)
        original={n:p.detach().cpu().clone() for n,p in trainable}
        torch.cuda.reset_peak_memory_stats()
        model.train()
        for step in range(args.steps):
            row=tokenized[step%len(tokenized)]
            batch={key:torch.tensor([value],device='cuda') for key,value in row.items()}
            batch['attention_mask']=torch.ones_like(batch['input_ids'])
            loss=model(**batch).loss
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0)
            if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            report['training_loss'].append(float(loss.detach().cpu()));save()
            print(json.dumps({'step':step+1,'loss':report['training_loss'][-1]}),flush=True)
        changed=sum(not torch.equal(original[n],p.detach().cpu()) for n,p in trainable)
        if not changed:raise ValueError('LoRA weights did not change')
        report['changed_trainable_tensors']=changed
        report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
        model.eval();model.gradient_checkpointing_disable()
        probe=torch.tensor([tokenized[0]['input_ids'][:32]],device='cuda')
        with torch.no_grad():signature=model(input_ids=probe,use_cache=False).logits[0,-1,:].float().cpu()
        adapter=args.out/'adapter';model.save_pretrained(adapter,safe_serialization=True);tok.save_pretrained(args.out/'tokenizer')
        del model,optimizer,trainable,loss,batch,original;gc.collect();torch.cuda.empty_cache()
        reloaded=PeftModel.from_pretrained(base(),adapter,is_trainable=False)
        reloaded.eval()
        with torch.no_grad():after=reloaded(input_ids=probe,use_cache=False).logits[0,-1,:].float().cpu()
        report['reload_max_logit_difference']=float((signature-after).abs().max())
        report['reload_equal']=bool(torch.allclose(signature,after,atol=1e-3,rtol=1e-3))
        if not report['reload_equal']:raise ValueError('Adapter reload logits differ')
        report['peak_process_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['adapter_sha256']={p.name:hash_file(p) for p in adapter.iterdir() if p.is_file()}
        report['status']='technical_smoke_completed'
    except BaseException as exc:
        report['status']='failed';report['error']=type(exc).__name__+': '+str(exc);raise
    finally:
        report['elapsed_seconds']=time.time()-started;save()

if __name__=='__main__':main()
