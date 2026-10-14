"""Native 9B NF4 three-step technical check, not a formal quality experiment."""
import argparse
import gc
import json
from pathlib import Path
import time

from lora_smoke import language_targets, hash_file
from semantic_lora_trainer import tokenize_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    report=dict(scope='native9B_NF4_three_steps_three_reviewed_train_rows_technical_only',
        status='starting',formal_training=False,quality_evaluation=False,losses=[],
        data_sha256=hash_file(args.data),script_sha256=hash_file(__file__))
    started=time.time()
    def save(stage):
        report['stage']=stage;report['elapsed_seconds']=time.time()-started
        (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        print(json.dumps(dict(stage=stage,status=report['status'],elapsed_seconds=report['elapsed_seconds'])),flush=True)
    save('starting')
    try:
        import resource
        import torch, transformers, peft, bitsandbytes
        from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
        torch.manual_seed(42)
        report['versions']={m.__name__:m.__version__ for m in (torch,transformers,peft,bitsandbytes)}
        report['gpu']=torch.cuda.get_device_name(0)
        rows=[json.loads(x) for x in args.data.read_text(encoding='utf8').splitlines()]
        if len(rows)!=3 or len({r['source_family_id'] for r in rows})!=3 or any(r['split']!='train' or r['review_status']!='reviewed' for r in rows):
            raise ValueError('only three reviewed distinct training families allowed')
        tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
        tokenized=tokenize_rows(rows,tok,2048)
        report['samples']=[dict(id=r['sample_id'],tokens=len(tokenized[r['sample_id']]['input_ids']),
            target_tokens=sum(x!=-100 for x in tokenized[r['sample_id']]['labels'])) for r in rows]
        config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        def base():
            return AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.bfloat16,
                device_map={'':'cuda:0'},local_files_only=True,trust_remote_code=False,
                attn_implementation='sdpa',quantization_config=config)
        save('loading_quantized_base')
        model=base()
        report['quantized_linear_modules']=sum(isinstance(m,bitsandbytes.nn.Linear4bit) for m in model.modules())
        if not getattr(model,'is_loaded_in_4bit',False) or not report['quantized_linear_modules']:
            raise ValueError('native model was not quantized')
        targets=language_targets([n for n,_ in model.named_modules()]);report['targets']=targets
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                             gradient_checkpointing_kwargs={'use_reentrant':False})
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,target_modules=targets,
                                            bias='none',task_type=TaskType.CAUSAL_LM))
        model.config.use_cache=False
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in n or 'language_model' not in n for n,p in trainable):
            raise ValueError('unexpected trainable base/vision/embedding parameters')
        report['trainable_parameters']=sum(p.numel() for _,p in trainable)
        original={n:p.detach().cpu().clone() for n,p in trainable}
        optimizer=torch.optim.AdamW([p for _,p in trainable],lr=1e-4)
        model.train();save('training')
        for i,row in enumerate(rows):
            batch={k:torch.tensor([v],device='cuda') for k,v in tokenized[row['sample_id']].items()}
            batch['attention_mask']=torch.ones_like(batch['input_ids'])
            loss=model(**batch).loss
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0)
            if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            report['losses'].append(float(loss.detach().cpu()));save('step_'+str(i+1))
        changed=sum(not torch.equal(original[n],p.detach().cpu()) for n,p in trainable)
        if not changed:raise ValueError('adapter did not update')
        report['changed_adapter_tensors']=changed
        model.eval();model.gradient_checkpointing_disable()
        probe=torch.tensor([tokenized[rows[0]['sample_id']]['input_ids'][:32]],device='cuda')
        with torch.no_grad():signature=model(input_ids=probe,use_cache=False).logits[0,-1].float().cpu()
        adapter=args.out/'adapter';model.save_pretrained(adapter,safe_serialization=True)
        report['adapter_sha256']={p.name:hash_file(p) for p in adapter.iterdir() if p.is_file()}
        del model,optimizer,trainable,loss,batch,original;gc.collect();torch.cuda.empty_cache()
        save('reloading')
        # k-bit preparation casts unquantized floating layers to float32. Reload
        # must use that same base precision before comparing adapter logits.
        reload_base=prepare_model_for_kbit_training(base(),use_gradient_checkpointing=False)
        reloaded=PeftModel.from_pretrained(reload_base,adapter,is_trainable=False);reloaded.eval()
        with torch.no_grad():after=reloaded(input_ids=probe,use_cache=False).logits[0,-1].float().cpu()
        report['reload_max_logit_difference']=float((after-signature).abs().max())
        report['reload_equal']=bool(torch.allclose(signature,after,atol=1e-3,rtol=1e-3))
        if not report['reload_equal']:raise ValueError('adapter reload differs')
        report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
        report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['status']='technical_passed'
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:save('finished')


if __name__=='__main__':main()
