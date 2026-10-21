"""Локальный веб-каталог Minecraft-скинов из skins-new-dataset/data.json."""

from __future__ import annotations

import argparse
import json
import mimetypes
import random
from collections import Counter, defaultdict
from itertools import combinations
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


class Catalog:
    def __init__(self, data_path: Path):
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        self.records = [
            {"filename": item["filename"], "tags": item["tags"]}
            for item in raw
            if isinstance(item.get("filename"), str) and isinstance(item.get("tags"), list)
        ]
        self.by_tag: dict[str, set[int]] = defaultdict(set)
        self.tag_counts: Counter[str] = Counter()
        for index, item in enumerate(self.records):
            for tag in item["tags"]:
                if isinstance(tag, str):
                    self.by_tag[tag].add(index)
                    self.tag_counts[tag] += 1
        # Матрица связей нужна только для популярных тегов: она строится один раз при запуске
        # и не раздувает память миллионами пар редких одноразовых меток.
        self.analysis_tags = [tag for tag, _ in self.tag_counts.most_common(60)]
        analysis_set = set(self.analysis_tags)
        self.cooccurrences: dict[str, Counter[str]] = defaultdict(Counter)
        for item in self.records:
            present = sorted({tag for tag in item["tags"] if tag in analysis_set})
            for left, right in combinations(present, 2):
                self.cooccurrences[left][right] += 1
                self.cooccurrences[right][left] += 1

    def popular_tags(self, limit: int) -> list[dict[str, int | str]]:
        return [{"tag": tag, "count": count} for tag, count in self.tag_counts.most_common(limit)]

    def related_tags(self, tag: str, limit: int) -> list[dict[str, int | str]]:
        return [{"tag": related, "count": count} for related, count in self.cooccurrences[tag].most_common(limit)]

    def find(self, query: str, selected_tags: list[str], limit: int) -> tuple[list[dict[str, object]], int]:
        selected_tags = [tag for tag in selected_tags if tag in self.by_tag]
        if selected_tags:
            matches = set.intersection(*(self.by_tag[tag] for tag in selected_tags))
        else:
            matches = set(range(len(self.records)))
        query = query.casefold().strip()
        if query:
            matches = {
                index for index in matches
                if query in self.records[index]["filename"].casefold()
                or any(query in tag.casefold() for tag in self.records[index]["tags"])
            }
        total = len(matches)
        indexes = random.sample(tuple(matches), k=min(limit, total)) if total else []
        return [self.records[index] for index in indexes], total


class CatalogHandler(BaseHTTPRequestHandler):
    catalog: Catalog
    site_root: Path
    skins_root: Path

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.client_address[0]} - {format % args}")

    def send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path, cache_seconds: int = 3600) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", f"public, max-age={cache_seconds}")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/popular-tags":
            limit = min(max(int(parse_qs(parsed.query).get("limit", [30])[0]), 1), 100)
            self.send_json({"tags": self.catalog.popular_tags(limit), "skin_count": len(self.catalog.records),
                            "tag_count": len(self.catalog.tag_counts)})
            return
        if parsed.path == "/api/skins":
            params = parse_qs(parsed.query)
            limit = min(max(int(params.get("limit", [48])[0]), 1), 100)
            tags = [tag for value in params.get("tags", []) for tag in value.split(",") if tag]
            skins, total = self.catalog.find(params.get("q", [""])[0], tags, limit)
            self.send_json({"skins": skins, "total": total})
            return
        if parsed.path == "/api/tag-relations":
            params = parse_qs(parsed.query)
            tag = params.get("tag", [""])[0]
            limit = min(max(int(params.get("limit", [24])[0]), 1), 60)
            self.send_json({"tag": tag, "related": self.catalog.related_tags(tag, limit)})
            return
        if parsed.path.startswith("/skins/"):
            relative = unquote(parsed.path.removeprefix("/skins/")).replace("\\", "/")
            candidate = (self.skins_root / relative).resolve()
            if not candidate.is_relative_to(self.skins_root.resolve()) or candidate.suffix.lower() != ".png":
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self.send_file(candidate, cache_seconds=86400)
            return
        relative = "index.html" if parsed.path in ("", "/") else parsed.path.lstrip("/")
        candidate = (self.site_root / relative).resolve()
        if not candidate.is_relative_to(self.site_root.resolve()):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        # HTML/JS/CSS не кэшируем: при правках сайта достаточно обычного обновления страницы.
        self.send_file(candidate, cache_seconds=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить локальный каталог Minecraft-скинов")
    parser.add_argument("--data", default="skins-new-dataset/data.json")
    parser.add_argument("--skins", default="skins-new-dataset/Skins")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    data_path, skins_path = Path(args.data).resolve(), Path(args.skins).resolve()
    if not data_path.is_file() or not skins_path.is_dir():
        raise SystemExit("Не найдены data.json или папка Skins. Проверьте --data и --skins.")
    CatalogHandler.catalog = Catalog(data_path)
    CatalogHandler.site_root = (Path(__file__).parent / "catalog_site").resolve()
    CatalogHandler.skins_root = skins_path
    server = ThreadingHTTPServer(("127.0.0.1", args.port), CatalogHandler)
    print(f"Каталог: http://127.0.0.1:{args.port}")
    print(f"Скинов: {len(CatalogHandler.catalog.records):,}; уникальных тегов: {len(CatalogHandler.catalog.tag_counts):,}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
