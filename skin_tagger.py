"""Обучение и применение multi-label классификатора тегов Minecraft-скинов."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.transforms import v2
from tqdm import tqdm


DEFAULT_DATA = Path("namemc-skins-dataset-main")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Проверьте установку PyTorch CUDA и драйвер NVIDIA.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def valid_skin_file(path: Path) -> bool:
    """Проверяет, что PNG можно полностью прочитать и это именно скин 64×64."""
    try:
        with Image.open(path) as image:
            image.verify()  # Проверка структуры PNG, CRC и целостности потока.
        # После verify объект PIL использовать нельзя, поэтому открываем картинку ещё раз.
        with Image.open(path) as image:
            return image.size == (64, 64)
    except Exception:
        # Повреждённые PNG могут выбрасывать разные исключения Pillow в зависимости от места повреждения.
        return False


def load_records(data_dir: Path, min_tag_count: int, verify_images: bool) -> tuple[list[tuple[Path, list[str]]], list[str], Counter]:
    raw = json.loads((data_dir / "data.json").read_text(encoding="utf-8"))
    cleaned: list[tuple[Path, list[str]]] = []
    all_tags: Counter = Counter()
    skins_dir = data_dir / "skins"
    # Одно перечисление каталога намного быстрее 194 тысяч отдельных stat-вызовов на Windows.
    available_filenames = {path.name for path in skins_dir.glob("*.png")}
    skipped_empty_tags = skipped_bad_filename = skipped_missing_file = skipped_invalid_image = 0

    for item in raw:
        tags = sorted({str(tag).strip() for tag in item.get("tags", []) if str(tag).strip()})
        if not tags:
            skipped_empty_tags += 1
            continue
        filename = item.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            skipped_bad_filename += 1
            continue
        image_path = skins_dir / filename
        # Пустые теги, null filename и отсутствующие картинки не могут быть обучающими примерами.
        if filename not in available_filenames:
            skipped_missing_file += 1
            continue
        if verify_images and not valid_skin_file(image_path):
            skipped_invalid_image += 1
            continue
        cleaned.append((image_path, tags))
        all_tags.update(tags)

    print(
        "Пропущено: "
        f"пустых тегов — {skipped_empty_tags:,}; "
        f"некорректных filename — {skipped_bad_filename:,}; "
        f"отсутствующих файлов — {skipped_missing_file:,}; "
        f"повреждённых или не-64x64 PNG — {skipped_invalid_image:,}"
    )

    labels = sorted(tag for tag, count in all_tags.items() if count >= min_tag_count)
    allowed = set(labels)
    records = [(path, [tag for tag in tags if tag in allowed]) for path, tags in cleaned]
    records = [(path, tags) for path, tags in records if tags]
    if not records or not labels:
        raise RuntimeError("После фильтрации не осталось обучающих примеров.")
    return records, labels, all_tags


class SkinDataset(Dataset):
    def __init__(self, records: list[tuple[Path, list[str]]], tag_to_index: dict[str, int], transform: nn.Module):
        self.records = records
        self.tag_to_index = tag_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, tags = self.records[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            x = self.transform(image)
        y = torch.zeros(len(self.tag_to_index), dtype=torch.float32)
        y[[self.tag_to_index[tag] for tag in tags]] = 1.0
        return x, y


def transforms(training: bool) -> nn.Module:
    # Скины уже имеют размер 64x64. Не делаем crop/resize: важны конкретные пиксели развёртки.
    layers: list[nn.Module] = []
    if training:
        layers.append(v2.RandomHorizontalFlip(p=0.5))
    layers.extend([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
    return v2.Compose(layers)


class SkinTagger(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.backbone = resnet18(weights=None, num_classes=num_tags)
        # Стандартный stem ResNet рассчитан на 224px; для нативного 64px он теряет слишком много деталей.
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, threshold: float) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    batches = 0
    true_positive = false_positive = false_negative = 0
    amp_enabled = device.type == "cuda"
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        predictions = torch.sigmoid(logits) >= threshold
        truth = targets.bool()
        true_positive += (predictions & truth).sum().item()
        false_positive += (predictions & ~truth).sum().item()
        false_negative += (~predictions & truth).sum().item()
        loss_sum += loss.item()
        batches += 1
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return loss_sum / max(batches, 1), 2 * precision * recall / max(precision + recall, 1e-12)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    labels: list[str],
    args: argparse.Namespace,
    epoch: int,
    score: float,
    best_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
        "labels": labels, "epoch": epoch, "val_f1": score, "best_f1": best_score, "architecture": "resnet18_64px",
        "mean": MEAN, "std": STD, "args": vars(args),
    }
    # Замена файла атомарна: не оставит битый checkpoint при выключении ПК во время сохранения.
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"CUDA: {torch.cuda.get_device_name(0)} | PyTorch CUDA: {torch.version.cuda}")

    records, labels, counts = load_records(Path(args.data_dir), args.min_tag_count, args.verify_images)
    random.Random(args.seed).shuffle(records)
    split = int(len(records) * (1 - args.val_fraction))
    train_records, val_records = records[:split], records[split:]
    tag_to_index = {tag: i for i, tag in enumerate(labels)}
    print(f"Примеров: {len(records):,}; train: {len(train_records):,}; val: {len(val_records):,}; тегов: {len(labels):,}")

    workers = min(args.workers, max(0, (torch.get_num_threads() or 1) - 1))
    loader_options = {"batch_size": args.batch_size, "num_workers": workers, "pin_memory": device.type == "cuda"}
    if workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(SkinDataset(train_records, tag_to_index, transforms(True)), shuffle=True, drop_last=True, **loader_options)
    val_loader = DataLoader(SkinDataset(val_records, tag_to_index, transforms(False)), shuffle=False, **loader_options)

    model = SkinTagger(len(labels)).to(device, memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format)
    # Усиливаем редкие метки, но ограничиваем вес, чтобы единичные случаи не дестабилизировали обучение.
    positives = torch.tensor([counts[tag] for tag in labels], dtype=torch.float32)
    pos_weight = ((len(train_records) - positives) / positives).sqrt().clamp(1, 10).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Состояние этого scheduler можно продолжать и при изменении целевого числа эпох.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -1.0
    start_epoch = 1
    last_checkpoint = Path(args.last_checkpoint)
    if args.resume and last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=True)
        if checkpoint.get("labels") != labels:
            raise RuntimeError(
                "Набор тегов в last.pt не совпадает с текущим. "
                "Используйте те же --min-tag-count и данные либо задайте --no-resume."
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint.get("best_f1", checkpoint.get("val_f1", -1.0)))
        print(f"Продолжаем с epoch {start_epoch}; последний checkpoint: {last_checkpoint}")
    if start_epoch > args.epochs:
        print(f"Обучение уже дошло до epoch {start_epoch - 1}. Увеличьте --epochs или используйте --no-resume.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for images, targets in bar:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step((epoch - 1) + (bar.n / len(train_loader)))
            running_loss += loss.item()
            bar.set_postfix(loss=f"{running_loss / (bar.n + 1):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        val_loss, val_f1 = validate(model, val_loader, criterion, device, args.threshold)
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, micro-F1@{args.threshold}={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(Path(args.output), model, optimizer, scheduler, scaler, labels, args, epoch, val_f1, best_f1)
            print(f"Сохранена лучшая модель: {args.output}")
        save_checkpoint(last_checkpoint, model, optimizer, scheduler, scaler, labels, args, epoch, val_f1, best_f1)
        print(f"Сохранена последняя эпоха: {last_checkpoint}")


@torch.no_grad()
def predict(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    labels: list[str] = checkpoint["labels"]
    model = SkinTagger(len(labels)).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    with Image.open(args.image) as image:
        x = transforms(False)(image.convert("RGB")).unsqueeze(0).to(device)
    probabilities = torch.sigmoid(model(x))[0].cpu()
    candidates = torch.where(probabilities >= args.threshold)[0].tolist()
    candidates.sort(key=lambda i: probabilities[i].item(), reverse=True)
    candidates = candidates[:args.top_k]
    # Строго требуемый формат результата: JSON-массив строк.
    print(json.dumps([labels[i] for i in candidates], ensure_ascii=False))
    if args.show_scores:
        print(json.dumps({labels[i]: round(probabilities[i].item(), 4) for i in candidates}, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Теги для Minecraft-скинов (64x64)")
    sub = p.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train", help="обучить модель")
    train_p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    train_p.add_argument("--output", default="checkpoints/skin_tagger.pt", help="лучшая модель по validation F1")
    train_p.add_argument("--last-checkpoint", default="checkpoints/last.pt", help="checkpoint, обновляемый после каждой эпохи")
    train_p.add_argument("--epochs", type=int, default=12)
    train_p.add_argument("--batch-size", type=int, default=128, help="уменьшите до 64 при CUDA out of memory")
    train_p.add_argument("--workers", type=int, default=4)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--min-tag-count", type=int, default=1, help="1 — сохранить все непустые теги; 5 — отфильтровать очень редкие")
    train_p.add_argument("--val-fraction", type=float, default=0.1)
    train_p.add_argument("--threshold", type=float, default=0.35)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    train_p.add_argument("--verify-images", action=argparse.BooleanOptionalAction, default=True,
                         help="проверять целостность PNG и размер 64x64 перед обучением")
    train_p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                         help="автоматически продолжать с --last-checkpoint")
    predict_p = sub.add_parser("predict", help="предсказать теги")
    predict_p.add_argument("image")
    predict_p.add_argument("--model", default="checkpoints/skin_tagger.pt")
    predict_p.add_argument("--threshold", type=float, default=0.35)
    predict_p.add_argument("--top-k", type=int, default=10)
    predict_p.add_argument("--show-scores", action="store_true")
    predict_p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.command == "train":
        train(arguments)
    else:
        predict(arguments)
