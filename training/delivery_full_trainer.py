"""Hash-bound full-dataset LoRA; fixed final step, recoverable optimizer commits."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import time

try:
    from .formal_schedule import schedule, resume_windows, data_signature
    from .semantic_lora_trainer import tokenize_rows, atomic_checkpoint
    from .lora_smoke import hash_file, language_targets
except ImportError:
    from formal_schedule import schedule, resume_windows, data_signature
    from semantic_lora_trainer import tokenize_rows, atomic_checkpoint
    from lora_smoke import hash_file, language_targets

CONFIG = dict(seed=42, epochs=1, accumulation=8, microbatch=1, optimizer_steps=111,
              lr=1e-4, rank=8, alpha=16, dropout=.05, max_length=2560)


def verify_inputs(root, arm, release_name='release.json'):
    release_path = root/release_name
    release = json.loads(release_path.read_text())
    if release.get('scope') != 'delivery_full888_LoRA_fixed111' or release.get('training_release_allowed') is not True:
        raise ValueError('missing scoped release')
    if release['config'] != CONFIG or arm not in {'4b', '9b'}:
        raise ValueError('training configuration drift')
    for name, expected in release['files'].items():
        if hash_file(root/name) != expected:
            raise ValueError('release file drift: '+name)
    if release.get('checkpoint_parent_release_sha256'):
        parent = json.loads((root/'release.json').read_text())
        if hash_file(root/'release.json') != release['checkpoint_parent_release_sha256']:
            raise ValueError('parent release drift')
        for key in ['config', 'models', 'versions', 'evidence']:
            if release[key] != parent[key]:raise ValueError('memory-only revision changed '+key)
        changed = {n for n in release['files'] if release['files'][n] != parent['files'][n]}
        if changed != {'delivery_full_trainer.py'} or hash_file(root/'trainer_before_memory.py') != parent['files']['delivery_full_trainer.py']:
            raise ValueError('unexpected source revision')
    rows = [json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    if len(rows) != 888 or any(r['split'] != 'train' for r in rows):
        raise ValueError('only 888 train rows allowed')
    if any([m['role'] for m in r['messages']] != ['system', 'user', 'assistant'] or
           json.loads(r['messages'][-1]['content']) != r['target'] for r in rows):
        raise ValueError('supervision drift')
    plan = json.loads((root/'schedule.json').read_text())
    if plan != schedule(rows, seed=42, epochs=1, accumulation=8):
        raise ValueError('schedule drift')
    provenance = dict(release_sha256=hash_file(release_path), arm=arm,
                      data_sha256=hash_file(root/'records.jsonl'), schedule_sha256=hash_file(root/'schedule.json'))
    return release, rows, plan, provenance


def compatible_checkpoint(saved, current, parent_sha256=None):
    if saved == current:return True
    return bool(parent_sha256 and saved == dict(current, release_sha256=parent_sha256))


def reconcile_steps(steps, committed_step, record):
    if [s['step'] for s in steps] != list(range(1, len(steps)+1)):
        raise ValueError('nonsequential report')
    if committed_step == len(steps):
        return steps
    if committed_step == len(steps)+1 and record['step'] == committed_step:
        return steps + [record]  # Checkpoint committed before report write was interrupted.
    raise ValueError('unreconciled report/checkpoint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=['4b', '9b'], required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--release', default='release.json', choices=['release.json', 'release_memory_v1.json'])
    args = parser.parse_args(); root = args.root
    release, rows, plan, provenance = verify_inputs(root, args.arm, args.release)
    base = Path(release['models'][args.arm]['path'])
    for name, expected in release['models'][args.arm]['files'].items():
        if hash_file(base/name) != expected:
            raise ValueError('base/tokenizer drift: '+name)
    out = root/args.arm
    if args.resume:
        if args.resume.parent.resolve() != out.resolve():
            raise ValueError('resume outside owned run')
        report = json.loads((out/'report.json').read_text())
    else:
        out.mkdir(exist_ok=False)
        report = dict(status='starting', steps=[], config=CONFIG, provenance=provenance,
                      precision='BF16' if args.arm == '4b' else 'NF4_double_quant_BF16_compute',
                      train_rows=888, validation_used=False, holdout_used=False,
                      checkpoint_selection='fixed_final_step111', deterministic_algorithms=False)
    if not compatible_checkpoint(report['provenance'], provenance, release.get('checkpoint_parent_release_sha256')):
        raise ValueError('resume provenance drift')
    report['provenance'] = provenance
    report['runtime_memory_policy'] = release.get('runtime_memory_policy', 'original_allocator')
    started = time.time(); prior_elapsed = report.get('elapsed_seconds', 0)
    def save():
        report['elapsed_seconds'] = prior_elapsed + time.time()-started
        temp = out/'report.tmp'; temp.write_text(json.dumps(report, indent=2)); temp.replace(out/'report.json')
    save()
    try:
        import numpy as np
        import torch, transformers, peft, bitsandbytes
        from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
        versions = {m.__name__:m.__version__ for m in (np, torch, transformers, peft, bitsandbytes)}
        if versions != release['versions']:
            raise ValueError('environment drift')
        report['versions'] = versions
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        tok = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
        tokenized = tokenize_rows(rows, tok, 2560)
        report['target_tokens'] = sum(sum(t != -100 for t in r['labels']) for r in tokenized.values())
        report['max_total_tokens'] = max(len(r['input_ids']) for r in tokenized.values())
        if (report['target_tokens'], report['max_total_tokens']) != (138852, 2397):
            raise ValueError('measured token gate drift')
        def load_base(training):
            kwargs = dict(dtype=torch.bfloat16, device_map={'':'cuda:0'}, local_files_only=True,
                          trust_remote_code=False, attn_implementation='sdpa')
            if args.arm == '9b':
                kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
                    bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForImageTextToText.from_pretrained(base, **kwargs)
            if bool(getattr(model, 'is_loaded_in_4bit', False)) != (args.arm == '9b'):
                raise ValueError('base precision mismatch')
            if args.arm == '9b':
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training,
                                                       gradient_checkpointing_kwargs={'use_reentrant':False})
            return model
        model = load_base(True)
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
            target_modules=language_targets([n for n,_ in model.named_modules()]), bias='none', task_type=TaskType.CAUSAL_LM))
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
        if not parameters or any('lora_' not in n or 'language_model' not in n for n in parameters):
            raise ValueError('unexpected trainable parameters')
        report['trainable_parameters'] = sum(p.numel() for p in parameters.values())
        report['base_and_vision_frozen'] = True
        initial = {n:hashlib.sha256(p.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                   for n,p in parameters.items()}
        if args.resume and initial != report['initial_adapter_tensor_sha256']:
            raise ValueError('initialization drift')
        report['initial_adapter_tensor_sha256'] = initial
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:max(0, 1-step/111))
        cursor = 0
        if args.resume:
            side = json.loads(args.resume.with_suffix('.json').read_text())
            if hash_file(args.resume) != side['checkpoint_sha256']:
                raise ValueError('checkpoint hash mismatch')
            saved = torch.load(args.resume, map_location='cpu', weights_only=False)
            if not compatible_checkpoint(saved['provenance'], provenance, release.get('checkpoint_parent_release_sha256')):
                raise ValueError('checkpoint provenance drift')
            cursor = saved['committed_step']
            report['steps'] = reconcile_steps(report['steps'], cursor, saved['step_record'])
            with torch.no_grad():
                for n,p in parameters.items():p.copy_(saved['weights'][n])
            optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            rng = saved['rng']; random.setstate(rng['python']); np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state_all(rng['cuda']); del saved
        windows = resume_windows(plan, cursor, data_signature_value=data_signature(rows))['next_windows']
        model.train(); optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        report['status'] = 'training'; save()
        for window in windows:
            tick = time.time(); losses = []
            for sid in window['sample_ids']:
                batch = {k:torch.tensor([v], device='cuda') for k,v in tokenized[sid].items()}
                batch['attention_mask'] = torch.ones_like(batch['input_ids'])
                loss = model(**batch).loss
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                losses.append(float(loss.detach().cpu())); (loss/window['loss_divisor']).backward()
                del batch, loss
                if release.get('runtime_memory_policy') == 'empty_cache_each_microbatch_v1':torch.cuda.empty_cache()
            grad = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.)
            if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            record = dict(step=window['step'], sample_ids=window['sample_ids'], losses=losses,
                          gradient_norm=float(grad), compute_seconds=time.time()-tick)
            checkpoint = atomic_checkpoint(torch, dict(provenance=provenance, committed_step=window['step'],
                step_record=record, weights={n:p.detach().cpu().clone() for n,p in parameters.items()},
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=dict(python=random.getstate(),
                numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())), out, window['step'])
            report['steps'].append(record); report['last_checkpoint'] = checkpoint
            report['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved()
            save(); print(json.dumps(dict(arm=args.arm, step=window['step'], seconds=time.time()-tick)), flush=True)
        model.eval(); model.gradient_checkpointing_disable()
        probe = torch.tensor([tokenized[rows[0]['sample_id']]['input_ids'][:32]], device='cuda')
        with torch.no_grad():before = model(input_ids=probe, use_cache=False).logits[0,-1].float().cpu()
        model.save_pretrained(out/'adapter', safe_serialization=True)
        report['adapter_files_sha256'] = {p.name:hash_file(p) for p in (out/'adapter').iterdir() if p.is_file()}
        del model, parameters, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(load_base(False), out/'adapter', is_trainable=False); model.eval()
        with torch.no_grad():after = model(input_ids=probe, use_cache=False).logits[0,-1].float().cpu()
        report['reload_max_logit_difference'] = float((before-after).abs().max())
        report['reload_equal'] = bool(torch.allclose(before, after, atol=1e-3, rtol=1e-3))
        if not report['reload_equal']:raise ValueError('reload differs')
        report['status'] = 'trained_fixed_step111_reload_verified'
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__+': '+str(exc)); raise
    finally:save()


if __name__ == '__main__':main()
