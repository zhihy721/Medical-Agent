"""Reviewed-text LoRA trainer with committed optimizer-boundary checkpoints.

Technical mode uses three reviewed train rows, two epochs and accumulation two.
It tests shuffle, short windows and process recovery, not extraction quality.
Full training requires a separate, hash-bound release artifact.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

if __package__:
    from .formal_schedule import data_signature, resume_windows, schedule
    from .lora_smoke import language_targets, hash_file
else:
    from formal_schedule import data_signature, resume_windows, schedule
    from lora_smoke import language_targets, hash_file


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def select_rows(rows, technical, technical_count=3):
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate dataset')
    if any(r.get('review_status')!='reviewed' or not r.get('messages') for r in rows):
        raise ValueError('unreviewed or unserialized data')
    train=[r for r in rows if r['split']=='train']
    if technical:
        # Exercise the longest textual targets/contexts without training validation.
        if technical_count not in {3,9}:raise ValueError('bounded technical count must be 3 or 9')
        train=sorted(train,key=lambda r:(-len(json.dumps(r['messages'],ensure_ascii=False)),r['sample_id']))[:technical_count]
        if len(train)!=technical_count or len({r['source_family_id'] for r in train})!=technical_count:
            raise ValueError('technical test needs distinct reviewed train families')
    return train


def verify_release(path, provenance):
    if path is None:
        raise ValueError('formal training needs an explicit release artifact')
    release=json.loads(Path(path).read_text(encoding='utf-8'))
    if release.get('training_release_allowed') is not True or release.get('scope') not in {'technical_pilot_release','full_sft_release'}:
        raise ValueError('training release is not granted')
    for key in ('data_sha256','trainer_sha256','schedule_sha256','config_sha256','base_sha256'):
        if release.get(key)!=provenance[key]:
            raise ValueError('stale training release: '+key)


def checkpoint_provenance(current, saved):
    if current!=saved:
        raise ValueError('resume provenance mismatch')


def tokenize_rows(rows, tokenizer, max_length):
    result={}
    for row in rows:
        msgs=row['messages']
        if len(msgs)<2 or msgs[-1]['role']!='assistant':
            raise ValueError('missing assistant target')
        prefix=tokenizer.apply_chat_template(msgs[:-1],tokenize=True,add_generation_prompt=True,
                                            enable_thinking=False,return_dict=False)
        whole=tokenizer.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False,
                                           enable_thinking=False,return_dict=False)
        if whole[:len(prefix)]!=prefix or not len(prefix)<len(whole)<=max_length:
            raise ValueError(row['sample_id']+': assistant boundary or length failure; no truncation')
        result[row['sample_id']]=dict(input_ids=whole,labels=[-100]*len(prefix)+whole[len(prefix):])
    return result


def atomic_checkpoint(torch, payload, out, step):
    final=out/f'checkpoint_{step:04d}.pt'
    temp=out/f'.checkpoint_{step:04d}.writing'
    if final.exists() or temp.exists():
        raise FileExistsError(final)
    torch.save(payload,temp)
    os.replace(temp,final)
    # Manifest is the commit record. An orphan .pt is never selected automatically.
    manifest=dict(step=step,checkpoint=final.name,checkpoint_sha256=hash_file(final),
                  provenance=payload['provenance'])
    sidecar=final.with_suffix('.json')
    with sidecar.open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','data','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--technical',action='store_true')
    p.add_argument('--technical-rows',type=int,choices=[3,9],default=3)
    p.add_argument('--release',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--stop-after',type=int)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--max-length',type=int,default=2048)
    a=p.parse_args()
    if not 1<=a.max_length<=2048 or not 0<a.lr<=1e-3:
        raise ValueError('invalid length or learning rate')
    if a.technical:
        a.epochs,a.accumulation=2,2 if a.technical_rows==3 else 8
    rows=[json.loads(line) for line in a.data.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    train=select_rows(rows,a.technical,a.technical_rows)
    plan=schedule(train,seed=a.seed,epochs=a.epochs,accumulation=a.accumulation)
    end=a.stop_after if a.stop_after is not None else plan['expected_optimizer_steps']
    if not 1<=end<=plan['expected_optimizer_steps']:
        raise ValueError('invalid stop boundary')
    if not a.technical and a.release is None:
        raise ValueError('formal training needs an explicit release artifact')
    a.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='starting',scope='technical_trainer_validation_only' if a.technical else 'released_sft',
                steps=[],technical=a.technical,training_release_allowed=False,
                data_sha256=hash_file(a.data),train_rows=len(train),validation_rows_used_for_gradient=0)
    def save():
        (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    started=time.perf_counter()
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import numpy as np
        import torch
        import transformers,peft
        from transformers import AutoModelForImageTextToText,AutoTokenizer
        from peft import LoraConfig,get_peft_model,TaskType
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        torch.use_deterministic_algorithms(True)
        config=dict(epochs=a.epochs,accumulation=a.accumulation,seed=a.seed,lr=a.lr,
                    max_length=a.max_length,r=8,alpha=16,dropout=.05,precision='bf16',
                    loss_weighting='mean_per_example; window_divisor_actual_count',
                    cuda_cache_policy='reclaim_between_microbatches',
                    deterministic_algorithms=True)
        provenance=dict(data_sha256=hash_file(a.data),trainer_sha256=hash_file(__file__),
            schedule_sha256=hash_file(Path(__file__).with_name('formal_schedule.py')),
            lora_targets_source_sha256=hash_file(Path(__file__).with_name('lora_smoke.py')),
            config_sha256=digest(config),plan_sha256=digest(plan),
            base_sha256={f.name:hash_file(f) for f in a.model.iterdir() if f.suffix=='.safetensors' or f.name in {'config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json'}},
            versions={m.__name__:m.__version__ for m in (torch,transformers,peft)})
        if not a.technical:verify_release(a.release,provenance)
        report.update(config=config,provenance=provenance,schedule=plan)
        tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
        # Validate the entire reviewed corpus, even though technical gradients use three rows.
        tokenized=tokenize_rows(rows,tok,a.max_length)
        report['lengths']={sid:len(r['input_ids']) for sid,r in tokenized.items()}
        model=AutoModelForImageTextToText.from_pretrained(a.model,dtype=torch.bfloat16,
            device_map={'':'cuda:0'},local_files_only=True,trust_remote_code=False,attn_implementation='sdpa')
        targets=language_targets([n for n,_ in model.named_modules()])
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
            target_modules=targets,bias='none',task_type=TaskType.CAUSAL_LM))
        model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        parameters={n:v for n,v in model.named_parameters() if v.requires_grad}
        if not parameters or any('lora_' not in n or 'language_model' not in n for n in parameters):
            raise ValueError('unexpected trainable parameters')
        report['lora_targets']=targets
        report['trainable_parameters']=sum(v.numel() for v in parameters.values())
        optimizer=torch.optim.AdamW(list(parameters.values()),lr=a.lr)
        total=plan['expected_optimizer_steps']
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:max(0,1-step/total))
        def rng():
            return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        cursor=0
        if a.resume:
            sidecar=json.loads(a.resume.with_suffix('.json').read_text(encoding='utf-8'))
            checkpoint_provenance(provenance,sidecar['provenance'])
            if sidecar['checkpoint']!=a.resume.name or sidecar['checkpoint_sha256']!=hash_file(a.resume):
                raise ValueError('checkpoint identity or hash mismatch')
            # Only load the hash-verified checkpoint authored by this same trusted program.
            saved=torch.load(a.resume,map_location='cpu',weights_only=False)
            checkpoint_provenance(provenance,saved['provenance'])
            cursor=saved['committed_step']
            if cursor!=sidecar['step'] or cursor>=end or set(saved['weights'])!=set(parameters):
                raise ValueError('resume cursor or weight set mismatch')
            with torch.no_grad():
                for n,v in parameters.items():v.copy_(saved['weights'][n])
            optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            state=saved['rng']
            random.setstate(state['python']);np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
            report['resume_source']=dict(path=str(a.resume),sha256=sidecar['checkpoint_sha256'],committed_step=cursor)
            del saved
        remaining=resume_windows(plan,cursor,data_signature_value=data_signature(train))['next_windows']
        model.train();optimizer.zero_grad(set_to_none=True)
        # Reclaim stale allocations from model construction and optimizer restoration.
        # This changes allocator cache management, not the objective or RNG stream.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for window in remaining:
            if window['step']>end:break
            before=time.perf_counter();losses=[]
            for sid in window['sample_ids']:
                example=tokenized[sid]
                batch={k:torch.tensor([example[k]],device='cuda') for k in ('input_ids','labels')}
                batch['attention_mask']=torch.ones_like(batch['input_ids'])
                loss=model(**batch).loss
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                losses.append(float(loss.detach()))
                (loss/window['loss_divisor']).backward()
                del loss,batch
                # Variable sequence shapes otherwise retained > physical VRAM in WSL.
                # Live LoRA gradients remain allocated; only unused allocator blocks go.
                torch.cuda.empty_cache()
            norm=torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            weights={n:v.detach().cpu().clone() for n,v in parameters.items()}
            committed=atomic_checkpoint(torch,dict(provenance=provenance,committed_step=window['step'],
                weights=weights,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng()),a.out,window['step'])
            observation=dict(step=window['step'],epoch=window['epoch'],sample_ids=window['sample_ids'],
                loss_divisor=window['loss_divisor'],losses=losses,gradient_norm=float(norm),
                lr=scheduler.get_last_lr()[0],seconds=time.perf_counter()-before,checkpoint=committed)
            report['steps'].append(observation);save();print(json.dumps({k:observation[k] for k in ('step','epoch','sample_ids','losses','seconds')}),flush=True)
            del weights
        report.update(status='technical_passed' if a.technical else 'trained',committed_step=report['steps'][-1]['step'],
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        report['device_total_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
        report['allocator_reserved_within_physical']=report['peak_gpu_reserved_bytes']<=report['device_total_memory_bytes']
        if not report['allocator_reserved_within_physical']:
            raise ValueError('allocator reserved peak exceeds physical GPU memory; do not release training')
        if not a.technical and end==total:
            model.save_pretrained(a.out/'adapter',safe_serialization=True);tok.save_pretrained(a.out/'adapter')
            report['adapter_exported']=True
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;save()


if __name__=='__main__':main()
