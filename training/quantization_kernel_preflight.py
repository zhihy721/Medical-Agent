"""NF4 CUDA kernel and adapter gradient check; no model/data quality claim."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('refuse to overwrite kernel evidence')
    report = dict(scope='random_512_linear_NF4_and_adapter_gradients_not_9B_training',
                  formal_training=False, quality_evaluation=False, status='starting')
    started = time.time()
    try:
        import torch
        import bitsandbytes as bnb
        torch.manual_seed(42)
        report.update(torch=torch.__version__, cuda=torch.version.cuda, bitsandbytes=bnb.__version__,
                      gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
        layer = bnb.nn.Linear4bit(512, 512, bias=False, compute_dtype=torch.bfloat16,
                                  compress_statistics=True, quant_type='nf4').cuda()
        layer.weight.requires_grad_(False)
        a = torch.nn.Parameter(torch.randn(8, 512, device='cuda', dtype=torch.bfloat16)*0.01)
        b = torch.nn.Parameter(torch.zeros(512, 8, device='cuda', dtype=torch.bfloat16))
        original = b.detach().clone()
        optimizer = torch.optim.AdamW([a, b], lr=1e-3)
        losses, grads = [], []
        for _ in range(2):
            x = torch.randn(1, 32, 512, device='cuda', dtype=torch.bfloat16)
            y = layer(x) + (x @ a.T) @ b.T
            loss = y.float().square().mean()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_([a, b], 1)
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise ValueError('nonfinite kernel loss/gradient')
            losses.append(float(loss)); grads.append(float(grad))
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        report.update(losses=losses, gradients=grads, adapter_changed=not torch.equal(original, b),
                      quantized_weight_dtype=str(layer.weight.dtype), quant_type=layer.weight.quant_state.quant_type,
                      compressed_statistics=bool(layer.weight.quant_state.nested),
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        if not report['adapter_changed']:
            raise ValueError('adapter did not update')
        report['status'] = 'kernel_passed'
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.time()-started
        args.out.write_text(json.dumps(report, indent=2), encoding='utf8')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
