"""Local skin generation UI. Model inference runs in a single background subprocess."""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import secrets
import subprocess
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

from PIL import Image, UnidentifiedImageError
from next_tag import DEFAULT_MODEL, NextTagPredictor

ROOT = Path(__file__).resolve().parent


def classic_to_slim(source: Path, destination: Path):
    """Remap all six faces of both arms and sleeves to the three-pixel UV layout."""
    with Image.open(source) as image:
        original = image.convert('RGBA')
    if original.size != (64, 64):
        raise ValueError('Expected a 64x64 skin')
    output = original.copy()
    for x, y in [(40, 16), (40, 32), (32, 48), (48, 48)]:
        output.paste((0, 0, 0, 0), (x, y, x + 16, y + 16))
        # (source x, source y, width, height, destination x, destination y, new width)
        faces = [(4, 0, 4, 4, 4, 0, 3), (8, 0, 4, 4, 7, 0, 3),
                 (0, 4, 4, 12, 0, 4, 4), (4, 4, 4, 12, 4, 4, 3),
                 (8, 4, 4, 12, 7, 4, 4), (12, 4, 4, 12, 11, 4, 3)]
        for sx, sy, width, height, dx, dy, new_width in faces:
            face = original.crop((x + sx, y + sy, x + sx + width, y + sy + height))
            face = face.resize((new_width, height), Image.Resampling.NEAREST)
            output.paste(face, (x + dx, y + dy))
    output.save(destination)


class GeneratorApp:
    def __init__(self, checkpoint, manifest, output, device='auto', next_tag_model=DEFAULT_MODEL,
                 tagger_model=ROOT / 'checkpoints/skin_tagger.pt'):
        self.checkpoint, self.output, self.device = checkpoint, output, device
        self.output.mkdir(parents=True, exist_ok=True)
        data = json.loads(manifest.read_text(encoding='utf-8'))
        self.tags = sorted([{'tag': tag, 'count': data['counts'].get(tag, 0)}
                            for tag in data['labels']], key=lambda row: (-row['count'], row['tag']))
        self.allowed = {row['tag'] for row in self.tags}
        self.models = {'default': {'name': 'Модель запуска', 'checkpoint': checkpoint, 'manifest': manifest,
                                   'tags': self.tags, 'allowed': self.allowed}}
        self.default_model = 'default'
        for model_id, name in [('generator_large_new_dataset', 'Generator Large · latest')]:
            directory = ROOT / 'checkpoints' / model_id
            if (directory / 'dataset.json').is_file():
                self.models[model_id] = {'name': name, 'checkpoint': directory / 'last.pt',
                                        'manifest': directory / 'dataset.json'}
                if (directory / 'last.pt').resolve() == checkpoint.resolve():
                    self.default_model = model_id
        if self.default_model != 'default':
            del self.models['default']
        self.model_lock = threading.Lock()
        self.jobs = {}
        self.lock = threading.Lock()
        self.active = None
        self.next_tag_path = Path(next_tag_model)
        self.suggestion_lock = threading.Lock()
        self.predictor = None
        self.predictor_signature = None
        self.suggestion_cache = {}
        self.tagger_path = Path(tagger_model)
        self.tagger_lock = threading.Lock()
        self.tagger = None
        self.tagger_signature = None

    def get_model(self, model_id=None):
        model_id = model_id or self.default_model
        if not isinstance(model_id, str) or model_id not in self.models:
            raise ValueError('Неизвестная модель генерации.')
        model = self.models[model_id]
        with self.model_lock:
            info = model['manifest'].stat()
            signature = (info.st_mtime_ns, info.st_size)
            if model.get('signature') != signature:
                data = json.loads(model['manifest'].read_text(encoding='utf-8'))
                model['tags'] = sorted([{'tag': tag, 'count': data['counts'].get(tag, 0)}
                                        for tag in data['labels']], key=lambda row: (-row['count'], row['tag']))
                model['allowed'] = {row['tag'] for row in model['tags']}
                model['signature'] = signature
        return model_id, model

    def list_models(self):
        return {'default_model': self.default_model,
                'models': [{'id': key, 'name': row['name'], 'ready': row['checkpoint'].is_file()}
                           for key, row in self.models.items()]}

    def describe_skin(self, content, model_id=None):
        _, selected_model = self.get_model(model_id)
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.format != 'PNG' or image.size != (64, 64):
                    raise ValueError('Загрузите скин PNG размером 64 × 64.')
                image.load()
                rgb = image.convert('RGB')
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError('Не удалось прочитать PNG. Проверьте файл скина.') from error
        with self.tagger_lock:
            import torch
            from skin_tagger import SkinTagger, transforms
            try:
                info = self.tagger_path.stat()
                signature = (info.st_mtime_ns, info.st_size)
                if self.tagger_signature != signature:
                    torch.set_num_threads(2)
                    checkpoint = torch.load(self.tagger_path, map_location='cpu', weights_only=True)
                    if checkpoint.get('architecture') != 'resnet18_64px':
                        raise ValueError('Нужен checkpoint первого распознавателя SkinTagger.')
                    model = SkinTagger(len(checkpoint['labels'])).eval()
                    model.load_state_dict(checkpoint['model_state'])
                    self.tagger = model
                    self.tagger_labels = checkpoint['labels']
                    self.tagger_signature = signature
            except Exception as error:
                raise RuntimeError('Не удалось загрузить распознаватель скинов: ' + str(error)) from error
            with torch.inference_mode():
                probabilities = self.tagger(transforms(False)(rgb).unsqueeze(0)).sigmoid()[0]
            candidates = (probabilities >= .35).nonzero().flatten().tolist()
            low_confidence = not candidates
            candidates = sorted(candidates, key=lambda i: float(probabilities[i]), reverse=True)[:10]
            if low_confidence:
                candidates = probabilities.topk(min(5, len(probabilities))).indices.tolist()
            return {'tags': [{'tag': self.tagger_labels[i].strip().casefold(),
                              'score': round(float(probabilities[i]), 4),
                              'supported': self.tagger_labels[i].strip().casefold() in selected_model['allowed']}
                             for i in candidates], 'low_confidence': low_confidence}

    def suggest(self, request):
        model_id, selected_model = self.get_model(request.get('model'))
        allowed = selected_model['allowed']
        tags = request.get('tags', [])
        if not isinstance(tags, list) or len(tags) > 32 or any(not isinstance(t, str) for t in tags):
            raise ValueError('Ожидается список не более 32 тегов.')
        names = sorted({tag.strip().casefold() for tag in tags})
        unknown = set(names) - allowed
        if unknown:
            raise ValueError('Неизвестные теги: ' + ', '.join(sorted(unknown)))
        with self.suggestion_lock:
            try:
                info = self.next_tag_path.stat()
                signature = (info.st_mtime_ns, info.st_size)
                if signature != self.predictor_signature:
                    import torch
                    torch.set_num_threads(2)
                    self.predictor = NextTagPredictor(self.next_tag_path)
                    self.predictor_signature = signature
                    self.suggestion_cache.clear()
            except (OSError, ValueError, RuntimeError) as error:
                raise RuntimeError('Не удалось загрузить модель следующего тега: ' + str(error)) from error
            key = (model_id, selected_model['signature'], tuple(names))
            if key not in self.suggestion_cache:
                context = [tag for tag in names if tag in self.predictor.indices]
                results = self.predictor.suggest(context, 8, allowed=allowed - set(names))
                if len(self.suggestion_cache) >= 128:
                    self.suggestion_cache.pop(next(iter(self.suggestion_cache)))
                self.suggestion_cache[key] = results
            return {'tags': self.suggestion_cache[key], 'model_epoch': self.predictor.epoch}

    def start(self, request):
        model_id, selected_model = self.get_model(request.get('model'))
        if not selected_model['checkpoint'].is_file():
            raise ValueError('Модель ещё не сохранила checkpoint. Дождитесь первых шагов обучения.')
        tags = request.get('tags')
        if not isinstance(tags, list) or not 1 <= len(tags) <= 32 or any(not isinstance(t, str) for t in tags):
            raise ValueError('Выберите от 1 до 32 тегов из подсказок.')
        tags = sorted({tag.strip().casefold() for tag in tags})
        unknown = set(tags) - selected_model['allowed']
        if unknown:
            raise ValueError('Неизвестные теги: ' + ', '.join(sorted(unknown)))
        steps = request.get('steps', 100)
        if type(steps) is not int or not 1 <= steps <= 1000:
            raise ValueError('Количество шагов должно быть от 1 до 1000.')
        with self.lock:
            if self.active:
                raise RuntimeError('Генерация уже идёт. Дождитесь её завершения.')
            job_id = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f') + '_' + secrets.token_hex(3)
            job = {'id': job_id, 'state': 'running', 'tags': tags, 'steps': steps,
                   'results': [], 'created_at': datetime.now().astimezone().isoformat(),
                   'generator_model': model_id, 'checkpoint': str(selected_model['checkpoint'].resolve())}
            self.jobs[job_id] = job
            self.active = job_id
        threading.Thread(target=self.run, args=(job_id,), daemon=True).start()
        return job.copy()

    def snapshot(self, job_id):
        with self.lock:
            return dict(self.jobs[job_id]) if job_id in self.jobs else None

    def run(self, job_id):
        staging = self.output / '.jobs' / job_id
        staging.mkdir(parents=True, exist_ok=True)
        job = self.snapshot(job_id)
        try:
            command = [sys.executable, '-u', str(ROOT / 'skin_generator.py'), 'generate',
                       '--checkpoint', job['checkpoint'], '--tags', json.dumps(job['tags']),
                       '--sampling-steps', str(job['steps']), '--count', '4',
                       '--device', self.device, '--output', str(staging)]
            with (staging / 'generation.log').open('w', encoding='utf-8') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         cwd=ROOT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if process.returncode:
                details = (staging / 'generation.log').read_text(encoding='utf-8', errors='replace')
                raise RuntimeError('Ошибка модели: ' + details[-2000:])
            results = []
            for path in sorted(staging.glob('*.png')):
                metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                stamp = datetime.now().astimezone()
                name = f"{stamp.strftime('%Y%m%d_%H%M%S_%f')}_seed{metadata['seed']}.png"
                destination = self.output / name
                classic_to_slim(path, destination)
                metadata.update(model='slim', converted_from='classic', created_at=stamp.isoformat(),
                                generator_model=job['generator_model'])
                destination.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
                results.append({'name': name, 'url': '/skins/' + name, 'seed': str(metadata['seed']),
                                'tags': metadata['tags'], 'created_at': metadata['created_at'], 'model': 'slim'})
            if len(results) != 4:
                raise RuntimeError('Модель не создала все четыре скина. Проверьте журнал генерации.')
            with self.lock:
                self.jobs[job_id].update(state='done', results=results)
        except Exception as error:
            with self.lock:
                self.jobs[job_id].update(state='failed', error=str(error))
        finally:
            with self.lock:
                self.active = None

    def history(self):
        results = []
        for path in sorted(self.output.glob('*.json'), reverse=True)[:24]:
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                if not path.with_suffix('.png').is_file():
                    continue
                results.append({'name': path.with_suffix('.png').name,
                                'url': '/skins/' + path.with_suffix('.png').name,
                                'seed': str(data['seed']), 'tags': data['tags'],
                                'created_at': data['created_at'], 'model': 'slim'})
            except (ValueError, KeyError):
                continue
        return results


class Handler(BaseHTTPRequestHandler):
    app: GeneratorApp

    def json_response(self, payload, status=200):
        content = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = unquote(parsed.path)
        model_id = parse_qs(parsed.query).get('model', [None])[0]
        if route == '/api/models':
            return self.json_response(self.app.list_models())
        if route == '/api/tags':
            try:
                key, model = self.app.get_model(model_id)
                return self.json_response({'tags': model['tags'], 'model': key})
            except ValueError as error:
                return self.json_response({'error': str(error)}, 400)
        if route == '/api/history':
            return self.json_response({'results': self.app.history()})
        if route.startswith('/api/jobs/'):
            job = self.app.snapshot(route.split('/')[-1])
            return self.json_response(job or {'error': 'Задание не найдено'}, 200 if job else 404)
        if route.startswith('/skins/'):
            root, name = self.app.output.resolve(), route.removeprefix('/skins/')
            path = (root / name).resolve()
            if path.parent != root or path.suffix != '.png':
                return self.send_error(403)
        else:
            root = (ROOT / 'generator_site').resolve()
            path = (root / ('index.html' if route == '/' else route.lstrip('/'))).resolve()
            if not path.is_relative_to(root):
                return self.send_error(403)
        if not path.is_file():
            return self.send_error(404)
        content = path.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route not in ('/api/generate', '/api/suggest', '/api/describe-skin'):
            return self.send_error(404)
        # The server is local; reject cross-origin browser submissions.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.json_response({'error': 'Недопустимый источник запроса'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if route == '/api/describe-skin':
                if not 0 < length <= 1024 * 1024:
                    raise ValueError('Выберите PNG размером до 1 МБ.')
                model_id = parse_qs(parsed.query).get('model', [None])[0]
                return self.json_response(self.app.describe_skin(self.rfile.read(length), model_id))
            if not 0 < length <= 8192:
                raise ValueError('Неверный размер запроса')
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError('Ожидается JSON-объект')
            if route == '/api/suggest':
                self.json_response(self.app.suggest(request))
            else:
                self.json_response(self.app.start(request), 202)
        except (ValueError, TypeError) as error:
            self.json_response({'error': str(error)}, 400)
        except RuntimeError as error:
            self.json_response({'error': str(error)}, 409)


def main():
    parser = argparse.ArgumentParser(description='Local tag-to-skin generator website')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/generator_large_new_dataset/last.pt')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'checkpoints/generator_large_new_dataset/dataset.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'generated_skins_web')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--next-tag-model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--tagger-model', type=Path, default=ROOT / 'checkpoints/skin_tagger.pt')
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.manifest.is_file():
        raise SystemExit('Не найден checkpoint или dataset.json. Проверьте --checkpoint и --manifest.')
    Handler.app = GeneratorApp(args.checkpoint.resolve(), args.manifest.resolve(), args.output.resolve(), args.device,
                              args.next_tag_model.resolve(), args.tagger_model.resolve())
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Генератор скинов: http://127.0.0.1:{args.port}', flush=True)
    print(f'Папка результатов: {args.output.resolve()}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
