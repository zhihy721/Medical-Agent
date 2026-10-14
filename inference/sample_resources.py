"""Sample GPU and WSL memory on the inference host during a benchmark.

Run this inside WSL, not through a new SSH session for every sample. The
result is a time series; a single nvidia-smi reading is not a peak.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


GPU_QUERY = ('nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits')
MEMINFO = Path('/proc/meminfo')


def parse_gpu_csv(value):
    rows = []
    for line in value.splitlines():
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split(',')]
        if len(cells) != 4 or any(not cell.isdecimal() for cell in cells):
            raise ValueError('unexpected nvidia-smi CSV row')
        index, used, total, utilization = map(int, cells)
        if used > total or utilization > 100:
            raise ValueError('invalid GPU memory or utilization')
        rows.append(dict(index=index, memory_used_mib=used,
                         memory_total_mib=total, utilization_percent=utilization))
    if not rows or len({row['index'] for row in rows}) != len(rows):
        raise ValueError('missing or duplicate GPU rows')
    return rows


def parse_meminfo(value):
    fields = {}
    for line in value.splitlines():
        key, _, rest = line.partition(':')
        if key not in {'MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'}:
            continue
        parts = rest.split()
        if len(parts) != 2 or parts[1] != 'kB' or not parts[0].isdecimal():
            raise ValueError('unexpected /proc/meminfo field: ' + key)
        fields[key] = int(parts[0]) * 1024
    if len(fields) != 4 or fields['MemAvailable'] > fields['MemTotal'] or fields['SwapFree'] > fields['SwapTotal']:
        raise ValueError('incomplete or invalid /proc/meminfo')
    return {key.lower()+'_bytes': value for key, value in fields.items()}


def sample_once():
    gpu = subprocess.run(GPU_QUERY, capture_output=True, text=True,
                         timeout=8, check=True)
    return dict(time_utc=datetime.now(timezone.utc).isoformat(),
                gpu=parse_gpu_csv(gpu.stdout),
                wsl=parse_meminfo(MEMINFO.read_text(encoding='ascii')))


def summarize(samples):
    good = [row for row in samples if row['status'] == 'ok']
    indices = sorted({gpu['index'] for row in good for gpu in row['gpu']})
    return dict(samples=len(samples), succeeded=len(good), failed=len(samples)-len(good),
                gpu_peak_used_mib={str(i): max(gpu['memory_used_mib'] for row in good
                    for gpu in row['gpu'] if gpu['index'] == i) for i in indices},
                wsl_min_mem_available_bytes=min((row['wsl']['memavailable_bytes']
                                                  for row in good), default=None),
                wsl_min_swap_free_bytes=min((row['wsl']['swapfree_bytes']
                                             for row in good), default=None))


def run(out, *, seconds, interval, stop_file=None):
    if seconds <= 0 or interval <= 0:
        raise ValueError('seconds and interval must be positive')
    if not MEMINFO.exists():
        raise RuntimeError('run the resource sampler inside Linux/WSL')
    out = Path(out)
    stop_file = Path(stop_file) if stop_file else None
    if stop_file and stop_file.exists():
        raise ValueError('stop file already exists')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(scope='inference_host_resource_sampling', status='running',
                  seconds=seconds, interval_seconds=interval,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gpu_query=list(GPU_QUERY), samples_file='samples.jsonl',
                  started_at_utc=datetime.now(timezone.utc).isoformat(),
                  stop_file=str(stop_file) if stop_file else None)
    path = out/'report.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    rows = []
    start = time.monotonic()
    next_sample = start
    try:
        with (out/'samples.jsonl').open('x', encoding='utf-8') as stream:
            while True:
                if rows and stop_file and stop_file.exists():
                    report['stop_reason'] = 'owned_benchmark_stop_marker'
                    break
                wait = next_sample - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                now = time.monotonic()
                if now - start > seconds and rows:
                    report['stop_reason'] = 'duration_limit'
                    break
                try:
                    row = dict(status='ok', **sample_once())
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    row = dict(status='failed', time_utc=datetime.now(timezone.utc).isoformat(),
                               error=type(exc).__name__ + ': ' + str(exc)[:200])
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                next_sample += interval
        report.update(summarize(rows))
        report['status'] = 'completed' if report['failed'] == 0 else 'completed_with_failures'
    except BaseException:
        report['status'] = 'interrupted'
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    report = run(args.out, seconds=args.seconds, interval=args.interval, stop_file=args.stop_file)
    print(json.dumps({key: report[key] for key in ('status', 'samples', 'succeeded',
          'failed', 'gpu_peak_used_mib', 'wsl_min_mem_available_bytes')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
