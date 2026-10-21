"""Small permutation-invariant neural predictor of another skin tag."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'checkpoints/next_tag/model.pt'


class NextTagModel(nn.Module):
    def __init__(self, num_tags, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_tags + 1, embedding_dim, padding_idx=0)
        self.network = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, num_tags))

    def forward(self, tags):
        mask = (tags != 0).unsqueeze(-1)
        pooled = (self.embedding(tags) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.network(pooled)


def read_data(path, min_count):
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, list):
        raise ValueError('data.json must contain a list of skin records')
    rows = [tuple(sorted({tag.strip().casefold() for tag in row.get('tags', [])
                          if isinstance(tag, str) and tag.strip()})) for row in raw if isinstance(row, dict)]
    counts = Counter(tag for row in rows for tag in row)
    labels = sorted(tag for tag, count in counts.items() if count >= min_count)
    indices = {tag: i + 1 for i, tag in enumerate(labels)}
    rows = [tuple(indices[tag] for tag in row if tag in indices) for row in rows]
    rows = [row for row in rows if len(row) >= 2]
    # Identical tag sets stay in one split, preventing memorized validation sets.
    train, validation = [], []
    for row in rows:
        key = ','.join(str(index) for index in row).encode()
        (validation if int.from_bytes(hashlib.sha256(key).digest()[:4], 'big') % 20 == 0 else train).append(row)
    if not train or not validation:
        raise ValueError('Need enough multi-tag skins for training and validation')
    return labels, counts, train, validation


def make_batch(rows, rng, augment=False):
    contexts, targets = [], []
    for row in rows:
        target = rng.randrange(len(row))
        context = [tag for i, tag in enumerate(row) if i != target]
        if augment:
            rng.shuffle(context)
            context = context[:rng.randint(0, len(context))]
        contexts.append(context)
        targets.append(row[target] - 1)
    tags = torch.zeros(len(rows), max(1, max(map(len, contexts))), dtype=torch.long)
    for i, context in enumerate(contexts):
        tags[i, :len(context)] = torch.tensor(context, dtype=torch.long)
    return tags, torch.tensor(targets, dtype=torch.long)


def exclude_given(logits, tags):
    result = logits.clone()
    row_indices = torch.arange(len(tags)).unsqueeze(1).expand_as(tags)
    valid = tags > 0
    result[row_indices[valid], tags[valid] - 1] = -torch.inf
    return result


@torch.no_grad()
def evaluate(model, rows, batch_size, popularity):
    model.eval()
    rng = random.Random(2026)
    loss_sum = top1 = top5 = baseline5 = 0
    for start in range(0, len(rows), batch_size):
        tags, target = make_batch(rows[start:start + batch_size], rng)
        logits = exclude_given(model(tags), tags)
        loss_sum += nn.functional.cross_entropy(logits, target, reduction='sum').item()
        predictions = logits.topk(min(5, logits.shape[1]), dim=1).indices
        top1 += (predictions[:, 0] == target).sum().item()
        top5 += (predictions == target[:, None]).any(1).sum().item()
        baseline = exclude_given(popularity.expand(len(tags), -1), tags)
        baseline5 += (baseline.topk(min(5, baseline.shape[1]), dim=1).indices == target[:, None]).any(1).sum().item()
    return {'loss': loss_sum / len(rows), 'top1': top1 / len(rows), 'top5': top5 / len(rows),
            'popularity_top5': baseline5 / len(rows)}


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    labels, counts, training, validation = read_data(args.data, args.min_count)
    model = NextTagModel(len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    popularity = torch.tensor([counts[tag] for tag in labels], dtype=torch.float)
    args.output.mkdir(parents=True, exist_ok=True)
    parameters = sum(p.numel() for p in model.parameters())
    print(f'Data: {len(training)} train / {len(validation)} validation, {len(labels)} tags; '
          f'{parameters:,} parameters; CPU', flush=True)
    best_loss = float('inf')
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(training)
        total_loss = 0
        for start in range(0, len(training), args.batch_size):
            tags, target = make_batch(training[start:start + args.batch_size], rng, augment=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(exclude_given(model(tags), tags), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(tags)
        metrics = evaluate(model, validation, args.batch_size, popularity)
        row = {'epoch': epoch, 'train_loss': total_loss / len(training), **metrics,
               'elapsed_seconds': round(time.monotonic() - started, 2)}
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs} loss={row['train_loss']:.4f} val={metrics['loss']:.4f} "
              f"top1={metrics['top1']:.1%} top5={metrics['top5']:.1%} "
              f"popularity_top5={metrics['popularity_top5']:.1%}", flush=True)
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            checkpoint = {'architecture': 'mean_tag_embedding_mlp_v1', 'labels': labels,
                          'state': model.state_dict(), 'config': {'embedding_dim': 32, 'hidden_dim': 64},
                          'epoch': epoch, 'metrics': metrics, 'parameters': parameters,
                          'training_rows': len(training), 'validation_rows': len(validation),
                          'source': str(args.data.resolve()),
                          'source_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest()}
            temporary = args.output / 'model.pt.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / 'model.pt')
        (args.output / 'training.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(f'Saved: {(args.output / "model.pt").resolve()} (best validation loss)', flush=True)


class NextTagPredictor:
    def __init__(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('architecture') != 'mean_tag_embedding_mlp_v1':
            raise ValueError('Select a next-tag checkpoint')
        self.labels = checkpoint['labels']
        self.indices = {tag: i + 1 for i, tag in enumerate(self.labels)}
        self.epoch = checkpoint.get('epoch')
        self.model = NextTagModel(len(self.labels), **checkpoint['config']).eval()
        self.model.load_state_dict(checkpoint['state'])

    @torch.no_grad()
    def suggest(self, names, top_k=8, allowed=None):
        names = sorted(set(names))
        unknown = [tag for tag in names if tag not in self.indices]
        if unknown:
            raise ValueError('Unknown tags: ' + ', '.join(unknown))
        tags = torch.tensor([[self.indices[tag] for tag in names] or [0]])
        logits = exclude_given(self.model(tags), tags)[0]
        if allowed is not None:
            excluded = torch.tensor([tag not in allowed for tag in self.labels])
            logits[excluded] = -torch.inf
        available = int(torch.isfinite(logits).sum())
        if not available:
            return []
        values, ids = logits.softmax(0).topk(min(top_k, available))
        return [{'tag': self.labels[i], 'score': round(float(value), 6)}
                for value, i in zip(values.tolist(), ids.tolist())]


def predict(args):
    torch.set_num_threads(args.threads)
    predictor = NextTagPredictor(args.model)
    names = sorted({tag.strip().casefold() for tag in args.tags.split(',') if tag.strip()})
    results = predictor.suggest(names, args.top_k)
    if not results:
        raise ValueError('All known tags are already provided')
    print(json.dumps(results, ensure_ascii=False, indent=2))


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return result


def main():
    parser = argparse.ArgumentParser(description='Train/predict the next skin tag from existing tags')
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--data', type=Path, default=ROOT / 'namemc-skins-dataset-main/data.json')
    training.add_argument('--output', type=Path, default=DEFAULT_MODEL.parent)
    training.add_argument('--epochs', type=positive_int, default=5)
    training.add_argument('--batch-size', type=positive_int, default=512)
    training.add_argument('--min-count', type=positive_int, default=5)
    training.add_argument('--lr', type=float, default=.003)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--threads', type=positive_int, default=4)
    inference = commands.add_parser('predict')
    inference.add_argument('--tags', default='', help='Existing tags separated by commas')
    inference.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    inference.add_argument('--top-k', type=positive_int, default=10)
    inference.add_argument('--threads', type=positive_int, default=4)
    args = parser.parse_args()
    if args.command == 'train':
        if not 0 < args.lr < float('inf'):
            parser.error('--lr must be positive and finite')
        train(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
