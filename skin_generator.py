"""Tag-conditioned diffusion generator for Minecraft RGBA skins, trained locally."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import secrets
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "namemc-skins-dataset-main"
DEFAULT_RUN = ROOT / "checkpoints" / "generator_large_new_dataset"
LARGE_RUN = ROOT / "checkpoints" / "generator_large"
ARCHITECTURE = "tag_ddpm_rgba_v1"


def atomic_replace(temporary, destination):
    # Windows readers/antivirus can briefly deny replacing an open destination.
    for attempt in range(200):
        try:
            Path(temporary).replace(destination)
            return
        except PermissionError:
            if attempt == 199:
                raise
            time.sleep(.05)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    atomic_replace(temp, path)


def durable_checkpoint(path, payload):
    """Flush new weights to disk and retain the preceding complete checkpoint."""
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        previous = path.with_name(path.stem + '.previous.pt')
        previous_temp = previous.with_suffix('.pt.tmp')
        if previous_temp.exists():
            previous_temp.unlink()
        try:
            os.link(path, previous_temp)
        except OSError:
            with path.open('rb') as source, previous_temp.open('wb') as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
        atomic_replace(previous_temp, previous)
    atomic_replace(temporary, path)


def load_training_checkpoint(path):
    path = Path(path)
    candidates = [path, path.with_name(path.stem + '.previous.pt'),
                  path.with_name('best.pt'), path.with_name('best.previous.pt')]
    errors = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        try:
            payload = torch.load(candidate, map_location='cpu', weights_only=True)
            required = {'model_state', 'optimizer_state', 'config', 'labels', 'step', 'epoch', 'batch'}
            if not isinstance(payload, dict) or not required.issubset(payload):
                raise ValueError('Incomplete training checkpoint')
        except Exception as error:
            errors.append(f'{candidate.name}: {error}')
            print(f'Cannot read checkpoint {candidate.name}: {error}', flush=True)
            continue
        if candidate != path:
            print(f'Recovering from {candidate.name}, step={payload["step"]}', flush=True)
            if path.exists():
                quarantine = path.with_name(f'{path.stem}.corrupt.{time.time_ns()}.pt')
                atomic_replace(path, quarantine)
            temporary = path.with_suffix('.restore.tmp')
            with candidate.open('rb') as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            atomic_replace(temporary, path)
        return payload
    raise RuntimeError('No readable training checkpoint. ' + '; '.join(errors))


def uv_masks():
    """Classic 64x64 UV rectangles; force opaque base faces at export."""
    base = torch.zeros(64, 64, dtype=torch.bool)
    overlay = torch.zeros_like(base)

    def box(mask, x, y, width, height, depth):
        mask[y:y + depth, x + depth:x + depth + 2 * width] = True
        mask[y + depth:y + depth + height, x:x + 2 * (depth + width)] = True

    for args in [(0, 0, 8, 8, 8), (16, 16, 8, 12, 4), (0, 16, 4, 12, 4),
                 (40, 16, 4, 12, 4), (16, 48, 4, 12, 4), (32, 48, 4, 12, 4)]:
        box(base, *args)
    for args in [(32, 0, 8, 8, 8), (16, 32, 8, 12, 4), (0, 32, 4, 12, 4),
                 (40, 32, 4, 12, 4), (0, 48, 4, 12, 4), (48, 48, 4, 12, 4)]:
        box(overlay, *args)
    return base, base | overlay


def prepare(data_dir: Path, run_dir: Path, min_tag_count=5, workers=8):
    data_dir = data_dir.resolve()
    metadata = data_dir / "data.json"
    if not metadata.is_file() or not (data_dir / "skins").is_dir():
        raise ValueError(f"Expected data.json and skins/ in {data_dir}")
    contents = metadata.read_bytes()
    signature = {"data_sha256": hashlib.sha256(contents).hexdigest(),
                 "skins_mtime": (data_dir / "skins").stat().st_mtime_ns,
                 "data_dir": str(data_dir), "min_tag_count": min_tag_count, "version": 1}
    manifest_path = run_dir / "dataset.json"
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached.get("signature") == signature:
            print(f"Using verified manifest: {len(cached['records']):,} skins, {len(cached['labels']):,} tags", flush=True)
            return cached
    raw = json.loads(contents)
    available = {p.name for p in (data_dir / "skins").glob("*.png")}
    candidates, seen = [], set()
    skipped = Counter()
    for row in raw:
        name = row.get("filename")
        tags = sorted({t.strip().casefold() for t in row.get("tags", []) if isinstance(t, str) and t.strip()})
        if not tags:
            skipped["empty_tags"] += 1
        elif not isinstance(name, str) or Path(name).name != name or name not in available:
            skipped["missing_or_invalid_filename"] += 1
        elif name in seen:
            skipped["duplicate_filename"] += 1
        else:
            seen.add(name)
            candidates.append({"filename": name, "tags": tags})

    def verify(row):
        try:
            path = data_dir / "skins" / row["filename"]
            with Image.open(path) as image:
                if image.format != "PNG" or image.size != (64, 64):
                    return None
                image.verify()
            with Image.open(path) as image:
                image.convert("RGBA").load()
            return row
        except (OSError, ValueError, SyntaxError):
            return None

    records = []
    print(f"Verifying {len(candidates):,} tagged PNG files from {data_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(verify, candidates), 1):
            if result is None:
                skipped["invalid_png"] += 1
            else:
                records.append(result)
            if index % 5000 == 0:
                print(f"Verified {index:,}/{len(candidates):,}", flush=True)
    counts = Counter(tag for row in records for tag in row["tags"])
    labels = sorted(tag for tag, count in counts.items() if count >= min_tag_count)
    allowed = set(labels)
    for row in records:
        row["tags"] = [tag for tag in row["tags"] if tag in allowed]
    records = [row for row in records if row["tags"]]
    if len(records) < 2 or not labels:
        raise ValueError("Need at least two valid tagged skins; lower --min-tag-count if necessary")
    manifest = {"signature": signature, "labels": labels, "counts": dict(counts),
                "records": records, "skipped": dict(skipped)}
    write_json(manifest_path, manifest)
    print(f"Dataset ready: {len(records):,} skins, {len(labels):,} tags; skipped={dict(skipped)}", flush=True)
    return manifest


class Skins(Dataset):
    def __init__(self, data_dir, records, labels):
        self.directory, self.records = data_dir / "skins", records
        self.indices = {tag: index + 1 for index, tag in enumerate(labels)}
        self.mask = uv_masks()[1]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.directory / row["filename"]) as image:
            rgba = image.convert("RGBA")
            pixels = torch.frombuffer(bytearray(rgba.tobytes()), dtype=torch.uint8).reshape(64, 64, 4)
        x = pixels.permute(2, 0, 1).float() / 255
        x[:3] *= x[3:4]  # Transparent RGB cannot contaminate the training signal.
        x *= self.mask.unsqueeze(0)
        tags = torch.tensor([self.indices[t] for t in row["tags"]], dtype=torch.long)
        return x * 2 - 1, tags


def collate(batch):
    images, tags = zip(*batch)
    padded = torch.zeros(len(tags), max(len(row) for row in tags), dtype=torch.long)
    for index, row in enumerate(tags):
        padded[index, :len(row)] = row
    return torch.stack(images), padded


class ResBlock(nn.Module):
    def __init__(self, incoming, outgoing, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, outgoing)
        self.affine = nn.Linear(condition_dim, outgoing * 2)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.affine(F.silu(condition)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(h))


class TagUNet(nn.Module):
    def __init__(self, num_tags, channels=32):
        super().__init__()
        dim = channels * 4
        self.time_dim = dim
        self.tag_embedding = nn.Embedding(num_tags + 1, dim, padding_idx=0)
        self.null_condition = nn.Parameter(torch.zeros(dim))
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.tag_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.input = nn.Conv2d(4, channels, 3, padding=1)
        widths = [channels, channels * 2, channels * 4, channels * 4]
        self.down_blocks = nn.ModuleList([ResBlock(w, w, dim) for w in widths])
        self.downsamples = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1)
                                         for a, b in zip(widths, widths[1:])])
        self.middle = ResBlock(widths[-1], widths[-1], dim)
        self.up_blocks = nn.ModuleList([ResBlock(a + b, b, dim)
                                       for a, b in zip(reversed(widths[1:]), reversed(widths[:-1]))])
        self.output = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, 4, 3, padding=1))

    def forward(self, x, timesteps, tags):
        half = self.time_dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / (half - 1))
        angles = timesteps.float()[:, None] * frequencies[None, :]
        time_condition = self.time_mlp(torch.cat([angles.sin(), angles.cos()], dim=1))
        present = (tags != 0).unsqueeze(-1)
        tag_condition = (self.tag_embedding(tags) * present).sum(1) / present.sum(1).clamp(min=1)
        tag_condition = torch.where(present.any(1), tag_condition, self.null_condition[None])
        condition = time_condition + self.tag_mlp(tag_condition)
        h = self.input(x)
        skips = []
        for index, block in enumerate(self.down_blocks):
            h = block(h, condition)
            if index < len(self.downsamples):
                skips.append(h)
                h = self.downsamples[index](h)
        h = self.middle(h, condition)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], 1), condition)
        return self.output(h)


def schedule(steps, device):
    # Cosine alpha schedule for training and deterministic DDIM inference.
    times = torch.linspace(0, steps, steps + 1, device=device) / steps
    values = torch.cos((times + .008) / 1.008 * math.pi / 2).square()
    values = values / values[0].clone()
    betas = (1 - values[1:] / values[:-1]).clamp(.0001, .999)
    return torch.cumprod(1 - betas, dim=0)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(name)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(num_tags, channels, architecture=ARCHITECTURE):
    if architecture == ARCHITECTURE:
        return TagUNet(num_tags, channels)
    if architecture == 'tag_ddpm_rgba_v2_attention':
        from skin_generator_large import LargeTagUNet
        return LargeTagUNet(num_tags, channels)
    raise ValueError(f'Unknown generator architecture: {architecture}')


def loss_batch(model, images, tags, alpha, weight, drop_probability=0.1, generator=None):
    if getattr(model, 'prediction_type', 'epsilon') == 'v':
        from skin_generator_large import v_loss_batch
        return v_loss_batch(model, images, tags, alpha, weight, drop_probability, generator)
    times = torch.randint(len(alpha), (len(images),), device=images.device, generator=generator)
    noise = torch.randn(images.shape, device=images.device, generator=generator)
    a = alpha[times, None, None, None]
    noisy = a.sqrt() * images + (1 - a).sqrt() * noise
    if drop_probability:
        tags = tags.clone()
        tags[torch.rand(len(tags), device=tags.device) < drop_probability] = 0
    prediction = model(noisy, times, tags)
    return ((prediction.float() - noise).square() * weight).sum() / (len(images) * 4 * weight.sum())


@torch.no_grad()
def sample(model, tags, diffusion_steps, sampling_steps=50, guidance=2.5, seed=42,
           initial_image=None, strength=.5):
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((len(tags), 4, 64, 64), generator=rng, device=device)
    alpha = schedule(diffusion_steps, device)
    sequence = torch.linspace(diffusion_steps - 1, 0, sampling_steps, device=device).long().unique_consecutive()
    if initial_image is not None:
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("--strength must be between 0 and 1")
        initial_image = initial_image.to(device)
        if initial_image.shape != x.shape:
            raise ValueError("Initial image must match the sampling batch: N x 4 x 64 x 64")
        if strength == 0:
            return ((initial_image + 1) / 2).clamp(0, 1).cpu()
        sequence = sequence[-max(1, math.ceil(len(sequence) * strength)):]
        if strength < 1:
            a = alpha[sequence[0]]
            x = a.sqrt() * initial_image + (1 - a).sqrt() * x
    model.eval()
    for index, t in enumerate(sequence):
        times = t.expand(len(x))
        prediction = model(x, times, tags)
        if guidance != 1:
            unconditional = model(x, times, torch.zeros_like(tags))
            conditional = prediction
            prediction = unconditional + guidance * (prediction - unconditional)
            if getattr(model, 'prediction_type', 'epsilon') == 'v':
                conditional_std = conditional.std((1, 2, 3), keepdim=True)
                guided_std = prediction.std((1, 2, 3), keepdim=True).clamp(min=1e-6)
                prediction = .5 * prediction + .5 * prediction * conditional_std / guided_std
        a = alpha[t]
        if getattr(model, 'prediction_type', 'epsilon') == 'v':
            clean = (a.sqrt() * x - (1 - a).sqrt() * prediction).clamp(-1, 1)
        else:
            clean = ((x - (1 - a).sqrt() * prediction) / a.sqrt()).clamp(-1, 1)
        previous_a = alpha[sequence[index + 1]] if index + 1 < len(sequence) else x.new_tensor(1.)
        # Recompute epsilon after clipping so each DDIM step remains consistent.
        epsilon = (x - a.sqrt() * clean) / (1 - a).sqrt()
        x = previous_a.sqrt() * clean + (1 - previous_a).sqrt() * epsilon
    return ((x + 1) / 2).clamp(0, 1).cpu()


def export_skin(pixels, path):
    base, valid = uv_masks()
    alpha = pixels[3].clone()
    # Model learns premultiplied colors, so unpremultiply before PNG export.
    colors = (pixels[:3] / alpha.clamp(min=1 / 255)).clamp(0, 1)
    opacity = (alpha >= .5) & valid
    opacity[base] = True
    output = torch.cat([colors, opacity[None].float()])
    output[:3, ~opacity] = 0
    data = (output.permute(1, 2, 0) * 255).round().byte().contiguous().numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data).save(path)


def transfer_generator(model, ema, checkpoint, labels):
    """Start a new dataset run from EMA weights, remapping tags by name."""
    source = checkpoint.get('ema_state', checkpoint['model_state'])
    old_indices = {tag: index + 1 for index, tag in enumerate(checkpoint['labels'])}
    state = model.state_dict()
    for key, value in source.items():
        if key != 'tag_embedding.weight':
            state[key] = value
    embedding = state['tag_embedding.weight']
    embedding[0].copy_(source['tag_embedding.weight'][0])
    for index, tag in enumerate(labels, 1):
        if tag in old_indices:
            embedding[index].copy_(source['tag_embedding.weight'][old_indices[tag]])
    model.load_state_dict(state, strict=True)
    ema.load_state_dict(model.state_dict(), strict=True)


def train(args):
    seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    architecture = getattr(args, 'architecture', ARCHITECTURE)
    args.output.mkdir(parents=True, exist_ok=True)
    initial = None
    init_from = getattr(args, 'init_from', None)
    if init_from:
        if args.output.resolve() == init_from.resolve().parent:
            raise ValueError('--init-from requires a separate --output directory')
        initial = torch.load(init_from, map_location='cpu', weights_only=True)
        if initial.get('architecture') != architecture or initial['config']['channels'] != args.channels:
            raise ValueError('--init-from requires the same architecture and channels')
        if initial['config']['diffusion_steps'] != args.diffusion_steps:
            raise ValueError('--init-from requires the same diffusion steps')
    status_path = args.output / "status.json"
    write_json(status_path, {"state": "preparing", "pid": os.getpid(), "data_dir": str(args.data.resolve())})
    manifest = prepare(args.data, args.output, args.min_tag_count, args.prepare_workers)
    if initial is not None:
        # Retain old tag IDs and add new tags, preserving the existing conditioning.
        old_labels = initial['labels']
        old_set = set(old_labels)
        manifest['labels'] = old_labels + [tag for tag in manifest['labels'] if tag not in old_set]
        for tag in manifest['labels']:
            manifest['counts'].setdefault(tag, 0)
        write_json(args.output / 'dataset.json', manifest)
    labels = manifest["labels"]
    records = manifest["records"].copy()
    random.Random(args.seed).shuffle(records)
    if args.limit:
        records = records[:args.limit]
    split = max(1, min(len(records) - 1, int(len(records) * args.validation_fraction)))
    validation, training = records[:split], records[split:]
    common = dict(batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate,
                  pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    loader = DataLoader(Skins(args.data, training, labels), shuffle=True, **common)
    val_loader = DataLoader(Skins(args.data, validation, labels), shuffle=False, **common)
    model = create_model(len(labels), args.channels, architecture).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    alpha = schedule(args.diffusion_steps, device)
    weight = (.05 + .95 * uv_masks()[1].float()).to(device)[None, None]
    last = args.output / "last.pt"
    step, start_epoch, start_batch, best = 0, 0, 0, float("inf")
    if args.resume and last.exists():
        checkpoint = load_training_checkpoint(last)
        config = checkpoint["config"]
        if checkpoint.get('architecture') != architecture:
            raise ValueError('Resume requires the same architecture; use a separate --output directory')
        if checkpoint["labels"] != labels or checkpoint["dataset_signature"] != manifest["signature"]:
            raise ValueError("Dataset or vocabulary changed. Use a new --output directory.")
        keys = ("channels", "diffusion_steps", "seed", "limit", "batch_size", "validation_fraction", "amp", "lr")
        if any(config[key] != getattr(args, key) for key in keys):
            raise ValueError("Resume requires the same model, split, batch size and optimizer options; use a new --output")
        model.load_state_dict(checkpoint["model_state"])
        ema.load_state_dict(checkpoint["ema_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        step, start_epoch, start_batch, best = checkpoint["step"], checkpoint["epoch"], checkpoint["batch"], checkpoint["best_val_loss"]
        torch.set_rng_state(checkpoint["rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"Resumed step={step}, epoch={start_epoch + 1}, batch={start_batch}", flush=True)
    elif initial is not None:
        transfer_generator(model, ema, initial, labels)
        print(f"Fine-tuning from {init_from}: source step={initial['step']}, "
              f"{len(initial['labels'])} existing tags, {len(labels) - len(initial['labels'])} new tags; "
              "EMA weights, new optimizer and epoch counter", flush=True)
    del initial
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output / "config.json", config)
    metrics_path = args.output / "metrics.jsonl"
    started, last_loss = time.monotonic(), None
    print(f"Training {architecture} on {device}: {len(training):,} train / {len(validation):,} validation, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)

    def status(state, epoch, batch, **extra):
        payload = {"state": state, "pid": os.getpid(), "step": step, "epoch": epoch,
                   "batch": batch, "batches_per_epoch": len(loader), "loss": last_loss,
                   "train_skins": len(training), "validation_skins": len(validation), "tags": len(labels),
                   "device": str(device), "elapsed_seconds": round(time.monotonic() - started, 1),
                   "architecture": architecture, "parameters": sum(p.numel() for p in model.parameters()),
                   "weights_mib": round(sum(p.numel() for p in model.parameters()) * 4 / 1024 ** 2, 1),
                   "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1) if device.type == 'cuda' else None,
                   "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
        try:
            write_json(status_path, payload)
        except PermissionError:
            # Progress reporting must never abort valid optimizer work.
            print("Warning: status.json is locked by a reader; will retry on next update", flush=True)
        return payload

    def save(epoch, batch, path=last):
        payload = {"architecture": architecture, "config": config, "labels": labels,
                   "dataset_signature": manifest["signature"], "model_state": model.state_dict(),
                   "ema_state": ema.state_dict(), "optimizer_state": optimizer.state_dict(),
                   "scaler_state": scaler.state_dict(), "step": step, "epoch": epoch, "batch": batch,
                   "best_val_loss": best, "loss": last_loss, "rng_state": torch.get_rng_state(),
                   "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        durable_checkpoint(path, payload)
        status("training", epoch + 1, batch, checkpoint=str(path.resolve()), checkpoint_step=step)

    current_epoch, current_batch = start_epoch, start_batch
    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch = epoch
            # Epoch-specific shuffle makes mid-epoch resume reproducible without saving a permutation.
            loader.generator = torch.Generator().manual_seed(args.seed + epoch + 10000)
            loader.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
            model.train()
            for batch_index, (images, tags) in enumerate(loader):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                if (args.output / "STOP").exists() or (args.max_steps and step >= args.max_steps):
                    save(epoch, batch_index)
                    status("stopped" if (args.output / "STOP").exists() else "completed", epoch + 1, batch_index)
                    return
                images, tags = images.to(device, non_blocking=True), tags.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                    loss = loss_batch(model, images, tags, alpha, weight)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    continue  # Overflow is not an optimizer update.
                with torch.no_grad():
                    for average, parameter in zip(ema.parameters(), model.parameters()):
                        average.lerp_(parameter, 1 - args.ema_decay)
                step += 1
                current_batch = batch_index + 1
                last_loss = float(loss.detach())
                if step == 1 or step % args.log_every == 0:
                    payload = status("training", epoch + 1, current_batch)
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(payload) + "\n")
                    print(f"epoch={epoch + 1}/{args.epochs} batch={current_batch}/{len(loader)} "
                          f"step={step} loss={last_loss:.6f} elapsed={payload['elapsed_seconds']}s", flush=True)
                if step == 1 or step % args.save_every == 0:
                    save(epoch, current_batch)
            ema.eval()
            val_rng = torch.Generator(device=device).manual_seed(args.seed + 100000)
            val_sum, val_count = 0., 0
            with torch.no_grad():
                for val_index, (images, tags) in enumerate(val_loader):
                    if val_index >= args.validation_batches:
                        break
                    images, tags = images.to(device), tags.to(device)
                    loss = loss_batch(ema, images, tags, alpha, weight, 0, val_rng)
                    val_sum += float(loss) * len(images)
                    val_count += len(images)
            val_loss = val_sum / val_count
            if architecture == 'tag_ddpm_rgba_v2_attention':
                lookup = {tag: index + 1 for index, tag in enumerate(labels)}
                for preview_index, requested in enumerate((['girl', 'pink', 'cute'], ['boy', 'blue', 'hoodie'])):
                    preview_tags = [tag for tag in requested if tag in lookup] or labels[:3]
                    condition = torch.tensor([[lookup[tag] for tag in preview_tags]], device=device)
                    pixels = sample(ema, condition, args.diffusion_steps, min(100, args.diffusion_steps), 2.5, 42 + preview_index)[0]
                    preview_path = args.output / 'previews' / f'epoch_{epoch + 1:02d}_sample_{preview_index + 1}.png'
                    export_skin(pixels, preview_path)
                    write_json(preview_path.with_suffix('.json'), {'tags': preview_tags, 'epoch': epoch + 1,
                               'step': step, 'seed': 42 + preview_index, 'weights': 'ema', 'validation_loss': val_loss})
            improved = val_loss < best
            best = min(best, val_loss)
            save(epoch + 1, 0)
            if improved:
                save(epoch + 1, 0, args.output / "best.pt")
            payload = status("training", epoch + 1, len(loader), val_loss=val_loss, validation_examples=val_count)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload) + "\n")
            print(f"Validation objective loss={val_loss:.6f} (generation quality needs separate inspection)", flush=True)
        status("completed", args.epochs, len(loader))
    except KeyboardInterrupt:
        save(current_epoch, current_batch)
        status("stopped", current_epoch + 1, current_batch)
        print("Interrupted: checkpoint saved", flush=True)
    except Exception as error:
        if step:
            save(current_epoch, current_batch)
        status("failed", current_epoch + 1, current_batch, error=str(error))
        raise


def parse_tags(text):
    if text.lstrip().startswith("["):
        tags = json.loads(text)
        if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
            raise ValueError("JSON tags must be an array of strings")
    else:
        tags = text.split(",")
    return sorted({tag.strip().casefold() for tag in tags if tag.strip()})


def load_edit_skin(path):
    """Use the same premultiplied RGBA representation as the training dataset."""
    with Image.open(path) as image:
        if image.format != 'PNG' or image.size != (64, 64):
            raise ValueError("Input skin must be a 64x64 PNG (classic model)")
        pixels = torch.frombuffer(bytearray(image.convert('RGBA').tobytes()),
                                  dtype=torch.uint8).reshape(64, 64, 4)
    x = pixels.permute(2, 0, 1).float() / 255
    x[:3] *= x[3:4]
    x *= uv_masks()[1].unsqueeze(0)
    return (x * 2 - 1).unsqueeze(0)


def generate(args):
    editing = args.command == 'edit'
    initial_image = load_edit_skin(args.input) if editing else None
    if editing and (not math.isfinite(args.strength) or not 0 <= args.strength <= 1):
        raise ValueError("--strength must be between 0 and 1")
    device = choose_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") not in (ARCHITECTURE, 'tag_ddpm_rgba_v2_attention'):
        raise ValueError("Select a generator checkpoint, e.g. checkpoints/generator_large_new_dataset/last.pt")
    labels, config = checkpoint["labels"], checkpoint["config"]
    indices = {tag: index + 1 for index, tag in enumerate(labels)}
    names = parse_tags(args.tags)
    unknown = [tag for tag in names if tag not in indices]
    if unknown:
        raise ValueError(f"Unknown tags: {unknown}. See 'skin_generator.py tags --search ...'; use NameMC English tags.")
    if not names:
        raise ValueError("Provide at least one tag")
    if args.sampling_steps > config["diffusion_steps"]:
        raise ValueError("--sampling-steps cannot exceed training diffusion steps")
    model = create_model(len(labels), config["channels"], checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint["ema_state"] if args.weights == "ema" else checkpoint["model_state"])
    outputs = []
    args.output.mkdir(parents=True, exist_ok=True)
    # Generate individually to avoid memory spikes for a large --count.
    for index in range(args.count):
        seed = secrets.randbits(63) if args.seed is None else args.seed + index
        tags = torch.tensor([[indices[tag] for tag in names]], device=device)
        result = sample(model, tags, config["diffusion_steps"], args.sampling_steps, args.guidance, seed,
                        initial_image=initial_image, strength=args.strength if editing else .5)[0]
        prefix = f"edited_{args.input.stem}" if editing else 'skin'
        path = args.output / f"{prefix}_step{checkpoint['step']}_seed{seed}.png"
        export_skin(result, path)
        metadata = {"tags": names, "seed": seed,
                   "training_step": checkpoint["step"], "weights": args.weights,
                   "checkpoint": str(args.checkpoint.resolve()), "guidance": args.guidance,
                   "sampling_steps": args.sampling_steps, "model": "classic"}
        if editing:
            metadata.update(input_skin=str(args.input.resolve()), strength=args.strength)
        write_json(path.with_suffix('.json'), metadata)
        outputs.append(str(path.resolve()))
    print(json.dumps(outputs, ensure_ascii=False), flush=True)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return value


def parser():
    root = argparse.ArgumentParser(description="Train a diffusion model: tags -> Minecraft RGBA 64x64 PNG")
    sub = root.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train_parser.add_argument("--output", type=Path, default=LARGE_RUN)
    train_parser.add_argument("--epochs", type=positive_int, default=20)
    train_parser.add_argument("--max-steps", type=int, default=0, help="0 means no step cap")
    train_parser.add_argument("--limit", type=int, default=0, help="Only for small smoke tests; 0 = full data")
    train_parser.add_argument("--min-tag-count", type=positive_int, default=5)
    train_parser.add_argument("--batch-size", type=positive_int, default=8)
    train_parser.add_argument("--channels", type=positive_int, default=96)
    train_parser.add_argument('--architecture', choices=[ARCHITECTURE, 'tag_ddpm_rgba_v2_attention'], default='tag_ddpm_rgba_v2_attention')
    train_parser.add_argument("--diffusion-steps", type=positive_int, default=1000)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--validation-fraction", type=float, default=.05)
    train_parser.add_argument("--validation-batches", type=positive_int, default=32)
    train_parser.add_argument("--workers", type=int, default=2)
    train_parser.add_argument("--prepare-workers", type=positive_int, default=8)
    train_parser.add_argument("--cpu-threads", type=positive_int, default=4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    train_parser.add_argument("--amp", action="store_true", help="Optional FP16; GTX 1070 defaults to FP32")
    train_parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument('--init-from', type=Path, help='Start a separate dataset run from pretrained EMA weights')
    train_parser.add_argument("--ema-decay", type=float, default=.999)
    train_parser.add_argument("--log-every", type=positive_int, default=10)
    train_parser.add_argument("--save-every", type=positive_int, default=200)
    generation_options = argparse.ArgumentParser(add_help=False)
    generate_parser = generation_options
    generate_parser.add_argument("--tags", required=True, help='Comma-separated or JSON: Girl,Pink,Cute')
    generate_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "last.pt")
    generate_parser.add_argument("--count", type=positive_int, default=4)
    generate_parser.add_argument("--sampling-steps", type=positive_int, default=50)
    generate_parser.add_argument("--guidance", type=float, default=2.5)
    generate_parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    generate_parser.add_argument("--seed", type=int, default=None,
                                 help="Fixed seed for reproducibility; omitted = random seed for each skin")
    generate_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    generate_parser.add_argument("--cpu-threads", type=positive_int, default=4)
    plain_generate = sub.add_parser('generate', parents=[generation_options])
    plain_generate.add_argument('--output', type=Path, default=ROOT / 'generated_skins')
    edit_parser = sub.add_parser('edit', parents=[generation_options], help='Edit a 64x64 skin using tags')
    edit_parser.add_argument('--input', type=Path, required=True, help='Source 64x64 PNG skin')
    edit_parser.add_argument('--strength', type=float, default=.5, help='0 preserves input; 1 generates anew')
    edit_parser.add_argument('--output', type=Path, default=ROOT / 'edited_skins')
    tag_parser = sub.add_parser("tags")
    tag_parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    tag_parser.add_argument("--search", default="")
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    prepare_parser.add_argument("--min-tag-count", type=positive_int, default=5)
    prepare_parser.add_argument("--prepare-workers", type=positive_int, default=8)
    return root


def main():
    arguments = parser().parse_args()
    if arguments.command == "train":
        if arguments.channels % 8 or not 0 < arguments.validation_fraction < 1:
            raise ValueError("--channels must be divisible by 8; validation fraction must be between 0 and 1")
        if arguments.workers < 0 or arguments.limit == 1 or arguments.limit < 0 or arguments.max_steps < 0:
            raise ValueError("Invalid workers/limit/max-steps")
        if not 0 <= arguments.ema_decay < 1 or arguments.lr <= 0:
            raise ValueError("Invalid EMA decay or learning rate")
        try:
            train(arguments)
        except Exception as error:
            status_path = arguments.output / 'status.json'
            if status_path.is_file():
                current = json.loads(status_path.read_text(encoding='utf-8'))
                if current.get('state') == 'preparing' and current.get('pid') == os.getpid():
                    write_json(status_path, {**current, 'state': 'failed', 'error': str(error)})
            raise
    elif arguments.command in ("generate", "edit"):
        generate(arguments)
    elif arguments.command == "prepare":
        prepare(arguments.data, arguments.output, arguments.min_tag_count, arguments.prepare_workers)
    else:
        manifest = json.loads((arguments.run / "dataset.json").read_text(encoding="utf-8"))
        print(json.dumps({tag: manifest["counts"][tag] for tag in manifest["labels"]
                          if arguments.search.casefold() in tag}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
