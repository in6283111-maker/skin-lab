"""Download versioned, checksum-verified Skin Lab model files from Hugging Face."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_target(root, filename):
    path = (root / filename).resolve()
    if not path.is_relative_to((root / 'checkpoints').resolve()) or not filename.startswith('checkpoints/'):
        raise ValueError(f'Invalid model path: {filename}')
    return path


def download(repo, revision, destination, groups='all', force=False):
    from huggingface_hub import hf_hub_download, HfApi
    # Resolve once: all files must belong to the same commit even if main changes.
    commit = HfApi().model_info(repo, revision=revision).sha
    index_path = hf_hub_download(repo, 'models.json', revision=commit)
    index = json.loads(Path(index_path).read_text(encoding='utf-8'))
    selected = set(index['models']) if groups == 'all' else set(groups.split(','))
    unknown = selected - set(index['models'])
    if unknown:
        raise ValueError('Unknown models: ' + ', '.join(sorted(unknown)))
    for group in sorted(selected):
        for entry in index['models'][group]['files']:
            target = safe_target(destination, entry['path'])
            if target.exists():
                if sha256(target) == entry['sha256']:
                    print(f'Already verified: {entry["path"]}', flush=True)
                    continue
                if not force:
                    raise ValueError(f'{target} differs from the published model. Use another --output folder '
                                     'or --force only if you want to replace it.')
            cached = Path(hf_hub_download(repo, entry['path'], revision=commit))
            if cached.stat().st_size != entry['size'] or sha256(cached) != entry['sha256']:
                raise ValueError(f'Checksum mismatch: {entry["path"]}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.download')
            import shutil
            shutil.copyfile(cached, temporary)
            os.replace(temporary, target)
            print(f'Downloaded: {entry["path"]}', flush=True)
    print(f'Model revision: {commit}', flush=True)


def main():
    settings_path = ROOT / 'model_hub.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8')) if settings_path.exists() else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=settings.get('repo_id') or os.environ.get('SKIN_LAB_MODEL_REPO'))
    parser.add_argument('--revision', default=settings.get('revision') or 'main')
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--models', default='all', help='all or comma-separated model IDs')
    parser.add_argument('--force', action='store_true', help='Replace differing local weights')
    args = parser.parse_args()
    if not args.repo:
        parser.error('Specify --repo OWNER/REPO or set repo_id in model_hub.json')
    try:
        download(args.repo, args.revision, args.output, args.models, args.force)
    except Exception as error:
        parser.exit(1, f'Model download failed: {error}\n')


if __name__ == '__main__':
    main()
