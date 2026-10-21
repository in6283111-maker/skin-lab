"""Resume a random, conflict-checked gallery run and publish completed batches."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.catalog_prompts import conflicts, make_prompt


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def publish(source, repo):
    from tools.build_gallery import build
    build(source, ROOT / 'docs')
    if repo:
        from huggingface_hub import HfApi
        # Reuse hf auth login; no token enters logs, command arguments or source.
        result = HfApi().upload_file(repo_id=repo, path_or_fileobj=str(ROOT / 'docs/data.json'),
                                    path_in_repo='gallery/data.json',
                                    commit_message='Refresh generated skin catalog')
        print(f'Published catalog: {result}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--publish-every', type=int, default=100)
    parser.add_argument('--publish-repo', default='')
    parser.add_argument('--publish-only', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/generator_large_new_dataset/last.pt')
    parser.add_argument('--source', type=Path, default=ROOT / 'generated_skins_web')
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if min(args.count, args.steps, args.batch_size, args.publish_every) < 1:
        parser.error('Count, steps, batch size and publish interval must be positive')
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    args.source = args.source.resolve()
    args.source.mkdir(parents=True, exist_ok=True)
    if args.publish_only:
        publish(args.source, args.publish_repo)
        return
    lock_path = args.run / 'run.lock'
    try:
        lock = lock_path.open('x')
    except FileExistsError:
        parser.error('Run is locked. If the previous process crashed, remove run.lock before resuming.')
    with lock:
        lock.write(str(os.getpid()))
    status_path = args.run / 'status.json'
    started = time.time()
    previous_status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    status = {'state': 'loading', 'pid': os.getpid(), 'completed': 0, 'total': args.count,
              'published': previous_status.get('published', 0), 'started_at': datetime.now(timezone.utc).isoformat()}
    save_json(status_path, status)
    try:
        import torch
        from skin_generator import create_model, export_skin, sample
        from skin_generator_server import classic_to_slim
        torch.set_num_threads(4)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if args.checkpoint.parent.name != 'generator_large_new_dataset':
            raise ValueError('The public gallery run must use generator_large_new_dataset')
        if args.steps > checkpoint['config']['diffusion_steps']:
            raise ValueError('Sampling steps exceed the training schedule')
        labels = checkpoint['labels']
        indices = {tag: index + 1 for index, tag in enumerate(labels)}
        plan_path = args.run / 'plan.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding='utf-8'))
            if (plan['checkpoint'] != str(args.checkpoint.resolve()) or plan['steps'] != args.steps
                    or plan['count'] != args.count or plan['source'] != str(args.source)
                    or plan['training_step'] != checkpoint['step']):
                raise ValueError('Resume parameters/checkpoint differ from the saved plan')
        else:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            entries, seen, seen_tags = [], set(), set()
            attempts = 0
            while len(entries) < args.count:
                attempts += 1
                if attempts > args.count * 100:
                    raise ValueError('Not enough distinct compatible prompts for this vocabulary')
                seed = secrets.randbits(63)
                tags = make_prompt(labels)
                if seed in seen or tuple(tags) in seen_tags:
                    continue
                seen.add(seed)
                seen_tags.add(tuple(tags))
                entries.append({'seed': str(seed), 'tags': tags,
                                'filename': f'{stamp}_batch{len(entries):04d}_seed{seed}.png'})
            plan = {'checkpoint': str(args.checkpoint.resolve()), 'steps': args.steps, 'count': args.count,
                    'source': str(args.source), 'training_step': checkpoint['step'], 'entries': entries}
            save_json(plan_path, plan)
        for entry in plan['entries']:
            if conflicts(entry['tags']) or set(entry['tags']) - set(labels):
                raise ValueError('Saved plan contains incompatible or unsupported tags')
        print(f'Validated {len(plan["entries"])} compatible prompts, steps={args.steps}', flush=True)
        if args.plan_only:
            return
        def complete(entry):
            return (args.source / entry['filename']).is_file() and (args.source / entry['filename']).with_suffix('.json').is_file()
        pending = [entry for entry in plan['entries'] if not complete(entry)]
        status['completed'] = args.count - len(pending)
        save_json(status_path, status)
        model = create_model(len(labels), checkpoint['config']['channels'], checkpoint['architecture']).to(args.device)
        model.load_state_dict(checkpoint['ema_state'])
        diffusion_steps, training_step = checkpoint['config']['diffusion_steps'], checkpoint['step']
        del checkpoint
        classic_dir = args.run / 'classic'
        classic_dir.mkdir(exist_ok=True)
        completed_at_start = status['completed']
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            max_tags = max(len(entry['tags']) for entry in batch)
            tag_tensor = torch.tensor([[indices[tag] for tag in entry['tags']] + [0] * (max_tags - len(entry['tags']))
                                       for entry in batch], device=args.device)
            before = time.time()
            pixels = sample(model, tag_tensor, diffusion_steps, args.steps, 2.5,
                            seeds=[int(entry['seed']) for entry in batch])
            for entry, result in zip(batch, pixels):
                destination = args.source / entry['filename']
                classic = classic_dir / entry['filename']
                export_skin(result, classic)
                classic_to_slim(classic, destination)
                save_json(destination.with_suffix('.json'), {
                    'tags': entry['tags'], 'seed': int(entry['seed']), 'generator_model': 'generator_large_new_dataset',
                    'training_step': training_step, 'weights': 'ema', 'sampling_steps': args.steps,
                    'guidance': 2.5, 'model': 'slim', 'created_at': datetime.now(timezone.utc).isoformat()})
            status['completed'] += len(batch)
            elapsed = time.time() - started
            per_skin = elapsed / max(1, status['completed'] - completed_at_start)
            status.update(state='generating', elapsed_seconds=round(elapsed),
                          eta_seconds=round((args.count - status['completed']) * per_skin),
                          last_batch_seconds=round(time.time() - before, 2))
            save_json(status_path, status)
            print(f'{status["completed"]}/{args.count} | batch {status["last_batch_seconds"]}s | ETA {status["eta_seconds"] / 60:.1f} min', flush=True)
            if status['completed'] - status['published'] >= args.publish_every or status['completed'] == args.count:
                try:
                    publish(args.source, args.publish_repo)
                    status['published'] = status['completed']
                    status.pop('publication_error', None)
                except Exception as error:
                    status['publication_error'] = str(error)
                    print(f'Publication failed, will retry at the next batch: {error}', flush=True)
                save_json(status_path, status)
        if status['published'] != args.count:
            try:
                publish(args.source, args.publish_repo)
                status['published'] = args.count
                status.pop('publication_error', None)
            except Exception as error:
                status['publication_error'] = str(error)
        status['state'] = 'done' if status['published'] == args.count else 'awaiting_publication'
        save_json(status_path, status)
    except BaseException as error:
        status.update(state='failed', error=str(error))
        save_json(status_path, status)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
