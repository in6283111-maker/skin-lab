"""Create a public inference-only bundle without optimizer states or local paths."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def export(root, output):
    output = output.resolve()
    if output == root.resolve() or output in root.resolve().parents:
        raise ValueError('Export into a separate release directory, never the source workspace')
    for obsolete in ('generator', 'generator_large'):
        directory = (output / 'checkpoints' / obsolete).resolve()
        if not directory.is_relative_to(output):
            raise ValueError('Export directory escapes the release folder')
        if directory.is_dir():
            shutil.rmtree(directory)
    index = {'format_version': 1, 'models': {}}
    for name in ('generator_large_new_dataset', 'skin_tagger', 'next_tag'):
        source = root / 'checkpoints' / (f'{name}/last.pt' if name.startswith('generator') else
                                         'next_tag/model.pt' if name == 'next_tag' else 'skin_tagger.pt')
        if not source.is_file():
            continue
        original = torch.load(source, map_location='cpu', weights_only=True)
        if name.startswith('generator'):
            payload = {key: original[key] for key in ('architecture', 'labels', 'model_state', 'ema_state', 'step', 'epoch')}
            payload['config'] = {key: original['config'][key] for key in ('channels', 'diffusion_steps')}
            manifest = json.loads((source.parent / 'dataset.json').read_text(encoding='utf-8'))
            compact = {'labels': payload['labels'], 'counts': {tag: manifest['counts'].get(tag, 0) for tag in payload['labels']}}
            manifest_target = output / 'checkpoints' / name / 'dataset.json'
            manifest_target.parent.mkdir(parents=True, exist_ok=True)
            manifest_target.write_text(json.dumps(compact, ensure_ascii=False), encoding='utf-8')
        elif name == 'skin_tagger':
            payload = {key: original[key] for key in ('architecture', 'labels', 'model_state', 'epoch')}
        else:
            payload = {key: original[key] for key in ('architecture', 'labels', 'state', 'config', 'epoch', 'metrics', 'parameters')}
        destination = output / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, destination)
        # Check that public artifacts can be loaded with the safe tensor-only loader.
        torch.load(destination, map_location='cpu', weights_only=True)
        paths = [destination] + ([manifest_target] if name.startswith('generator') else [])
        entries = []
        for path in paths:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            entries.append({'path': path.relative_to(output).as_posix(), 'size': path.stat().st_size, 'sha256': digest.hexdigest()})
        index['models'][name] = {'architecture': payload['architecture'], 'epoch': payload['epoch'],
                                'step': payload.get('step'), 'tags': len(payload['labels']), 'files': entries}
        print(f'Exported {name}: {sum(row["size"] for row in entries) / 1024**2:.1f} MiB', flush=True)
        del original, payload
    (output / 'models.json').write_text(json.dumps(index, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    export(args.root, args.output)
