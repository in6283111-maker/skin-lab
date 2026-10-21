"""Преобразовать skins-new-dataset/data.json в реляционную SQLite-базу."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE skins (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    url TEXT
);
CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE
);
CREATE TABLE skin_tags (
    skin_id INTEGER NOT NULL REFERENCES skins(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (skin_id, tag_id)
) WITHOUT ROWID;
CREATE INDEX idx_skin_tags_tag_skin ON skin_tags(tag_id, skin_id);
CREATE INDEX idx_skins_filename ON skins(filename);
"""


def batches(rows: list[tuple], size: int = 5_000):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Создать SQLite-базу скинов и тегов из data.json")
    parser.add_argument("--input", default="skins-new-dataset/data.json")
    parser.add_argument("--output", default="skins-new-dataset/skins_catalog.sqlite3")
    parser.add_argument("--replace", action="store_true", help="заменить существующую базу после успешной сборки")
    args = parser.parse_args()

    input_path, output_path = Path(args.input).resolve(), Path(args.output).resolve()
    if not input_path.is_file():
        raise SystemExit(f"Не найден входной JSON: {input_path}")
    if output_path.exists() and not args.replace:
        raise SystemExit(f"База уже существует: {output_path}. Для пересборки добавьте --replace.")

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    tag_names: set[str] = set()
    for item in raw:
        filename = item.get("filename")
        if not isinstance(filename, str) or not filename:
            continue
        tags = sorted({tag.strip() for tag in item.get("tags", []) if isinstance(tag, str) and tag.strip()})
        records.append({"filename": filename, "url": item.get("url"), "tags": tags})
        tag_names.update(tags)
    print(f"JSON: {len(records):,} скинов; {len(tag_names):,} уникальных тегов.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    connection = sqlite3.connect(temporary_path)
    try:
        connection.executescript(SCHEMA)
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = NORMAL")
        with connection:
            connection.executemany("INSERT INTO tags(name) VALUES(?)", ((tag,) for tag in sorted(tag_names)))
            tag_ids = {name: tag_id for tag_id, name in connection.execute("SELECT id, name FROM tags")}
            skin_rows = [(item["filename"], item["url"] if isinstance(item["url"], str) else None) for item in records]
            for batch in batches(skin_rows):
                connection.executemany("INSERT OR IGNORE INTO skins(filename, url) VALUES(?, ?)", batch)
            skin_ids = {filename: skin_id for skin_id, filename in connection.execute("SELECT id, filename FROM skins")}
            links: list[tuple[int, int]] = []
            for item in records:
                skin_id = skin_ids[item["filename"]]
                links.extend((skin_id, tag_ids[tag]) for tag in item["tags"])
                if len(links) >= 20_000:
                    connection.executemany("INSERT OR IGNORE INTO skin_tags(skin_id, tag_id) VALUES(?, ?)", links)
                    links.clear()
            if links:
                connection.executemany("INSERT OR IGNORE INTO skin_tags(skin_id, tag_id) VALUES(?, ?)", links)
        connection.execute("ANALYZE")
        connection.commit()
    finally:
        connection.close()

    # Существующая БД будет заменена только полностью готовой временной базой.
    os.replace(temporary_path, output_path)
    with sqlite3.connect(output_path) as check:
        skins = check.execute("SELECT COUNT(*) FROM skins").fetchone()[0]
        tags = check.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        links = check.execute("SELECT COUNT(*) FROM skin_tags").fetchone()[0]
        integrity = check.execute("PRAGMA integrity_check").fetchone()[0]
    print(f"Готово: {output_path}\nskins={skins:,}; tags={tags:,}; skin_tags={links:,}; integrity={integrity}")


if __name__ == "__main__":
    main()
