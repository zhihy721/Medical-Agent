"""Remote, resumable official ModelScope downloads with a sealed file manifest.

Run in the download-only virtualenv; does not import torch or alter model serving.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time


MODELS = ('Qwen3.5-2B', 'Qwen3.5-9B')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(files):
    seen = set()
    for item in files:
        path = PurePosixPath(item['Path'])
        if path.is_absolute() or '..' in path.parts or str(path) in seen:
            raise ValueError('unsafe or duplicate model file')
        seen.add(str(path))
        if item.get('Type') != 'blob' or not item.get('Revision'):
            raise ValueError('missing per-file revision')
        digest = item.get('Sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('missing official SHA256')
        if not isinstance(item.get('Size'), int) or item['Size'] < 0:
            raise ValueError('invalid file size')
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json',
                'model.safetensors.index.json', 'LICENSE'}
    if not required <= seen or not any(p.endswith('.safetensors') for p in seen):
        raise ValueError('incomplete model manifest')


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    tmp.replace(path)


def run(out, model_root):
    os.environ.setdefault('MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS', '8')
    import modelscope
    from modelscope.hub.api import HubApi
    out, model_root = Path(out), Path(model_root)
    out.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    api = HubApi()
    results = []
    for model in MODELS:
        mid = 'Qwen/' + model
        dest = model_root / model
        manifest_path = out / (model + '.manifest.json')
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['model_id'] != mid:
                raise ValueError('existing manifest identity changed')
        else:
            files = api._api.legacy.list_repo_files(
                repo_id=mid, repo_type='model', revision='master', recursive=True)
            validate_manifest(files)
            manifest = dict(model_id=mid, source='https://modelscope.cn/models/' + mid,
                            branch_at_inventory='master', revision_policy='per_file_commit_and_sha256',
                            sdk_version=modelscope.__version__, files=files,
                            total_bytes=sum(f['Size'] for f in files), created_at=time.time())
            dump(manifest_path, manifest)
        validate_manifest(manifest['files'])
        dest.mkdir(exist_ok=True)
        verified = []
        for item in manifest['files']:
            progress = dict(model=model, file=item['Path'], stage='download',
                            verified_files=len(verified), total_files=len(manifest['files']), updated_at=time.time())
            dump(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            # Keep SDK single-stream partials resumable. New large files can use
            # its bounded parallel Range downloader when the size is supplied.
            partial = (dest / item['Path']).with_suffix(Path(item['Path']).suffix + '.incomplete')
            path = api._api.downloader.download_file(mid, 'model', item['Path'],
                revision=item['Revision'], local_dir=dest, expected_sha256=item['Sha256'],
                file_size=None if partial.exists() else item['Size'])
            actual = dict(path=item['Path'], bytes=path.stat().st_size, sha256=file_sha(path))
            if actual['bytes'] != item['Size'] or actual['sha256'] != item['Sha256']:
                raise ValueError('file integrity failed: ' + item['Path'])
            verified.append(actual)
        index = json.loads((dest / 'model.safetensors.index.json').read_text())
        shards = set(index['weight_map'].values())
        if not shards <= {f['path'] for f in verified}:
            raise ValueError('weight index references missing shards')
        result = dict(model_id=mid, directory=str(dest), verified=True, files=verified,
                      manifest_sha256=file_sha(manifest_path), weight_shards=sorted(shards),
                      completed_at=time.time())
        verified_path = out / (model + '.verified.json')
        if verified_path.exists():
            previous = json.loads(verified_path.read_text())
            for key in ('model_id', 'directory', 'files', 'manifest_sha256', 'weight_shards'):
                if previous[key] != result[key]:
                    raise ValueError('previous verified model changed: ' + model)
            result = previous
        else:
            dump(verified_path, result)
        results.append(result)
    dump(out / 'completed.json', dict(verified=True, models=results, completed_at=time.time()))
    dump(out / 'progress.json', dict(stage='completed', verified=True, updated_at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model-root', default=str(Path.home()/'models'))
    args = parser.parse_args()
    try:
        run(args.out, args.model_root)
    except Exception as exc:
        dump(Path(args.out) / 'failed.json', dict(error=type(exc).__name__, message=str(exc), at=time.time()))
        raise
