"""Пакетное присвоение тегов скинам той же моделью, что использует skin_tagger.py."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from skin_tagger import SkinTagger, choose_device, transforms


class FolderDataset(Dataset):
    def __init__(self, folder: Path):
        self.folder = folder
        self.paths: list[Path] = []
        # Список путей, а не сами изображения: для большого набора не расходуется память на пиксели.
        for current_root, _, filenames in os.walk(folder):
            root = Path(current_root)
            self.paths.extend(root / filename for filename in filenames if filename.lower().endswith(".png"))
        self.transform = transforms(False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            if image.size != (64, 64):
                raise ValueError(f"Скин должен быть 64x64: {path}")
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.relative_to(self.folder).as_posix()


def save_entry(file, entry: dict[str, object], first: bool) -> bool:
    if not first:
        file.write(",\n")
    json.dump(entry, file, ensure_ascii=False)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Проставить теги всем скинам папки моделью skin_tagger")
    parser.add_argument("folder", nargs="?", default="skins-new-dataset/Skins", help="папка со скинами")
    parser.add_argument("--model", default="checkpoints/skin_tagger.pt", help="checkpoint, полученный skin_tagger.py train")
    parser.add_argument("--output", default="skins-new-dataset/data.json", help="итоговый data.json")
    parser.add_argument("--batch-size", type=int, default=256, help="уменьшите до 128 при CUDA out of memory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.35, help="минимальная вероятность одного тега")
    parser.add_argument("--top-k", type=int, default=10, help="не более стольких тегов на скин")
    parser.add_argument("--min-tags", type=int, default=4, help="сохранить скин только при этом числе тегов; 4 означает «больше 3»")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.batch_size < 1 or args.workers < 0 or args.top_k < 1 or args.min_tags < 1:
        raise SystemExit("--batch-size, --top-k и --min-tags должны быть не меньше 1; --workers — не меньше 0")
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold должен быть в диапазоне от 0 до 1")

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"Папка не найдена: {folder}")
    output = Path(args.output).resolve()
    if output.is_relative_to(folder):
        raise SystemExit("data.json должен находиться вне папки со скинами.")

    device = choose_device(args.device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    labels: list[str] = checkpoint["labels"]
    model = SkinTagger(len(labels)).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        print(f"CUDA: {torch.cuda.get_device_name(0)} | тегов в модели: {len(labels):,}")

    dataset = FolderDataset(folder)
    if not len(dataset):
        raise SystemExit(f"В папке нет PNG: {folder}")
    loader_kwargs: dict[str, object] = {
        "batch_size": args.batch_size, "shuffle": False, "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    if args.workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, **loader_kwargs)
    print(f"Найдено PNG: {len(dataset):,}; сохраняются только скины с ≥{args.min_tags} тегами.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    kept = 0
    with temporary.open("w", encoding="utf-8") as file, torch.inference_mode():
        file.write("[\n")
        first = True
        for images, filenames in tqdm(loader, desc="Тегирование", unit="batch"):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                probabilities = torch.sigmoid(model(images))
            values, indices = torch.topk(probabilities, k=min(args.top_k, len(labels)), dim=1)
            for filename, row_values, row_indices in zip(filenames, values.cpu(), indices.cpu()):
                tags = [labels[index] for value, index in zip(row_values.tolist(), row_indices.tolist()) if value >= args.threshold]
                if len(tags) >= args.min_tags:
                    # Совместимо с форматом исходного data.json; URL для локальных файлов неизвестен.
                    first = save_entry(file, {"url": None, "tags": tags, "filename": filename}, first)
                    kept += 1
        file.write("\n]\n")
    # Atomic replace: прежний data.json не будет испорчен при аварийном завершении обработки.
    temporary.replace(output)
    print(f"Готово. Обработано: {len(dataset):,}; сохранено (≥{args.min_tags} тегов): {kept:,}.")
    print(f"Файл: {output}")


if __name__ == "__main__":
    main()
