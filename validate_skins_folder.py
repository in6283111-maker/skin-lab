"""Массовая проверка и очистка папки со скинами Minecraft 64×64."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterator

from PIL import Image
from tqdm import tqdm


def iter_png_files(root: Path) -> Iterator[Path]:
    """Рекурсивно отдаёт PNG, не собирая весь огромный каталог в память."""
    for current_root, _, filenames in os.walk(root):
        base = Path(current_root)
        for filename in filenames:
            if filename.lower().endswith(".png"):
                yield base / filename


def load_template_mask(path: Path) -> bytes:
    """Возвращает alpha-маску допустимых пикселей эталонного скина."""
    try:
        with Image.open(path) as image:
            if image.size != (64, 64):
                raise ValueError(f"размер {image.size[0]}x{image.size[1]}, нужен 64x64")
            image.verify()
        with Image.open(path) as image:
            alpha = image.convert("RGBA").getchannel("A")
            alpha.load()
            return alpha.tobytes()
    except Exception as error:
        raise SystemExit(f"Эталонный скин повреждён или некорректен: {path} ({type(error).__name__}: {error})") from error


def check_skin(path: Path, allowed_mask: bytes) -> tuple[Path, str | None]:
    """Возвращает причину брака или None, если файл — корректный скин 64×64."""
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                return path, "расширение .png, но внутренний формат не PNG"
            image.verify()  # Проверяет структуру PNG, CRC и поток сжатых данных.
        # verify закрывает возможность дальнейшего чтения текущего объекта, открываем файл повторно.
        with Image.open(path) as image:
            if image.size != (64, 64):
                return path, f"неверный размер {image.size[0]}x{image.size[1]}, нужен 64x64"
            rgba = image.convert("RGBA")
            rgba.load()  # Полностью декодирует изображение.
            skin_alpha = rgba.getchannel("A").tobytes()
            outside_pixels = sum(skin > 0 and template == 0 for skin, template in zip(skin_alpha, allowed_mask))
            if outside_pixels:
                return path, f"{outside_pixels} непрозрачных пикселей находятся вне alpha-маски эталона"
    except Exception as error:
        return path, f"повреждён или не читается: {type(error).__name__}: {error}"
    return path, None


def chunks(items: Iterator[Path], size: int) -> Iterator[list[Path]]:
    chunk: list[Path] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверить и очистить папку PNG-скинов Minecraft 64x64")
    parser.add_argument("folder", nargs="?", default="skins-new-dataset/Skins", help="папка со скинами")
    parser.add_argument("--delete", action="store_true", help="удалить невалидные файлы; без ключа выполняется только проверка")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="число потоков проверки")
    parser.add_argument("--log", default="invalid_skins.jsonl", help="файл журнала проблемных файлов")
    parser.add_argument("--template", default="skins-new-dataset/steve.png",
                        help="эталонный скин 64x64: его непрозрачные пиксели задают разрешённую область")
    args = parser.parse_args()

    root = Path(args.folder).resolve()
    if not root.is_dir():
        raise SystemExit(f"Папка не найдена: {root}")
    if args.workers < 1:
        raise SystemExit("--workers должен быть не меньше 1")

    template_path = Path(args.template).resolve()
    if template_path == root or not template_path.is_file():
        raise SystemExit(f"Эталонный скин не найден: {template_path}")
    allowed_mask = load_template_mask(template_path)
    allowed_pixels = sum(value > 0 for value in allowed_mask)

    log_path = Path(args.log).resolve()
    if log_path.is_relative_to(root):
        raise SystemExit("Журнал должен находиться вне проверяемой папки.")

    checked = valid = invalid = deleted = delete_errors = 0
    action = "УДАЛЕНИЕ" if args.delete else "ПРОВЕРКА БЕЗ УДАЛЕНИЯ"
    print(f"{action}: {root}\nЭталон: {template_path} ({allowed_pixels:,} допустимых пикселей); потоков: {args.workers}; журнал: {log_path}")

    # executor.map ограниченно забирает элементы из итератора, а chunks не даёт создать очередь на миллионы путей.
    with log_path.open("w", encoding="utf-8") as log_file, ThreadPoolExecutor(max_workers=args.workers) as executor:
        progress = tqdm(desc="Проверка", unit="skin")
        for batch in chunks(iter_png_files(root), 1_000):
            for path, reason in executor.map(partial(check_skin, allowed_mask=allowed_mask), batch):
                checked += 1
                progress.update(1)
                if reason is None:
                    valid += 1
                    continue
                invalid += 1
                entry = {"path": str(path), "reason": reason, "deleted": False}
                if args.delete:
                    try:
                        path.unlink()
                        deleted += 1
                        entry["deleted"] = True
                    except OSError as error:
                        delete_errors += 1
                        entry["delete_error"] = f"{type(error).__name__}: {error}"
                log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        progress.close()

    print(
        f"Готово. Проверено: {checked:,}; корректных: {valid:,}; "
        f"некорректных: {invalid:,}; удалено: {deleted:,}; ошибок удаления: {delete_errors:,}."
    )
    print(f"Подробности: {log_path}")
    if not args.delete and invalid:
        print("Это был dry-run. Для удаления запустите ту же команду с --delete.")


if __name__ == "__main__":
    main()
