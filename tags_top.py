"""List popular tags from a downloaded generator vocabulary."""
import argparse
import json
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parent / 'checkpoints/generator_large/dataset.json')
    parser.add_argument('--top', type=int, default=100)
    args = parser.parse_args()
    counts = json.loads(args.manifest.read_text(encoding='utf-8'))['counts']
    for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:args.top]:
        print(f'{tag}: {count}')
