"""Build a public, static GitHub Pages catalog from generated skin PNG/JSON pairs."""
import argparse
import base64
import io
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'generator_large_new_dataset': 'Generator Large · latest'}


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for path in sorted(source.glob('*.png'), reverse=True):
        metadata_path = path.with_suffix('.json')
        try:
            data = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
            model = data.get('generator_model')
            if model not in MODELS or model == 'unknown':
                # Only the known directory name is exported; never publish the local path.
                parts = str(data.get('checkpoint', '')).replace('\\', '/').split('/')
                candidate = parts[-2] if len(parts) >= 2 else None
                model = candidate if candidate in MODELS else 'unknown'
            if model != 'generator_large_new_dataset':
                continue
            with Image.open(path) as image:
                if image.format != 'PNG' or image.size != (64, 64):
                    raise ValueError('Not a 64x64 PNG')
                rgba = image.convert('RGBA')
                clean = Image.frombytes('RGBA', rgba.size, rgba.tobytes())
            buffer = io.BytesIO()
            clean.save(buffer, format='PNG', optimize=True)
            tags = sorted({tag.strip().casefold() for tag in data.get('tags', [])
                           if isinstance(tag, str) and tag.strip()})
            rows.append({'id': path.stem, 'filename': path.name, 'seed': str(data['seed']),
                         'tags': tags, 'generator': model, 'created_at': data.get('created_at', ''),
                         'layout': 'slim' if data.get('model') == 'slim' else 'default',
                         'training_step': data.get('training_step'),
                         'sampling_steps': data.get('sampling_steps'),
                         'image': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')})
        except (OSError, ValueError, KeyError) as error:
            skipped.append({'file': path.name, 'error': str(error)})
    counts = Counter(tag for row in rows for tag in row['tags'])
    payload = {'version': 1, 'models': MODELS, 'skins': rows,
               'popular_tags': [{'tag': tag, 'count': count} for tag, count in counts.most_common()]}
    (output / 'data.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    vendor = output / 'vendor'
    vendor.mkdir(exist_ok=True)
    for name in ('skinview3d.bundle.min.js', 'LICENSE.skinview3d'):
        shutil.copyfile(ROOT / 'generator_site/vendor' / name, vendor / name)
    (output / '.nojekyll').touch()
    print(json.dumps({'published_skins': len(rows), 'models': dict(Counter(row['generator'] for row in rows)),
                      'skipped': skipped, 'catalog_bytes': (output / 'data.json').stat().st_size}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'generated_skins_web')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs')
    args = parser.parse_args()
    build(args.source, args.output)
