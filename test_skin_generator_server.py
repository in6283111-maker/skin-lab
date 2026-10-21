import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import torch
from PIL import Image

from skin_generator import ARCHITECTURE, create_model
from skin_generator_server import GeneratorApp, Handler, classic_to_slim
from next_tag import NextTagModel


class WebsiteTests(unittest.TestCase):
    def test_model_suggestions_filter_vocabulary_cache_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'data.json'
            manifest.write_text(json.dumps({'labels': ['girl', 'pink', 'cute'],
                                           'counts': {'girl': 10, 'pink': 8, 'cute': 5}}))
            checkpoint = root / 'next.pt'
            model = NextTagModel(4)
            def save(epoch, preferred):
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                    model.network[-1].bias[3] = 100
                    model.network[-1].bias[preferred] = 5
                temporary = checkpoint.with_suffix('.tmp')
                torch.save({'architecture': 'mean_tag_embedding_mlp_v1', 'labels': ['girl', 'pink', 'cute', 'rare'],
                            'config': {'embedding_dim': 32, 'hidden_dim': 64},
                            'state': model.state_dict(), 'epoch': epoch}, temporary)
                temporary.replace(checkpoint)
            save(1, 1)
            app = GeneratorApp(root / 'unused.pt', manifest, root / 'outputs', next_tag_model=checkpoint)
            first = app.suggest({'tags': ['Girl']})
            self.assertEqual(first['tags'][0]['tag'], 'pink')
            self.assertEqual({row['tag'] for row in first['tags']}, {'pink', 'cute'})
            self.assertEqual(first, app.suggest({'tags': ['girl']}))
            save(2, 2)
            updated = app.suggest({'tags': ['girl']})
            self.assertEqual(updated['model_epoch'], 2)
            self.assertEqual(updated['tags'][0]['tag'], 'cute')
            with self.assertRaises(ValueError):
                app.suggest({'tags': ['unknown']})

    def test_slim_keeps_all_arm_faces_and_clears_unused_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new('RGBA', (64, 64), (10, 20, 30, 255))
            colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255),
                      (255, 255, 0, 255), (255, 0, 255, 255), (0, 255, 255, 255)]
            for x, y in [(40, 16), (40, 32), (32, 48), (48, 48)]:
                for box, color in zip([(4, 0, 8, 4), (8, 0, 12, 4), (0, 4, 4, 16),
                                       (4, 4, 8, 16), (8, 4, 12, 16), (12, 4, 16, 16)], colors):
                    image.paste(color, tuple(v + (x if i % 2 == 0 else y) for i, v in enumerate(box)))
            image.save(root / 'source.png')
            classic_to_slim(root / 'source.png', root / 'slim.png')
            with Image.open(root / 'slim.png') as slim:
                self.assertEqual(slim.getpixel((8, 8)), image.getpixel((8, 8)))
                for x, y in [(40, 16), (40, 32), (32, 48), (48, 48)]:
                    for point, color in zip([(4, 0), (7, 0), (0, 4), (4, 4), (7, 4), (11, 4)], colors):
                        self.assertEqual(slim.getpixel((x + point[0], y + point[1])), color)
                    self.assertEqual(slim.getpixel((x + 14, y + 4)), (0, 0, 0, 0))

    def test_http_generate_four_real_outputs_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.set_num_threads(2)
            model = create_model(2, 8, ARCHITECTURE)
            checkpoint = root / 'model.pt'
            torch.save(dict(architecture=ARCHITECTURE, labels=['pink', 'girl'], step=5,
                            config=dict(channels=8, diffusion_steps=20),
                            ema_state=model.state_dict(), model_state=model.state_dict()), checkpoint)
            manifest = root / 'dataset.json'
            manifest.write_text(json.dumps({'labels': ['pink', 'girl'], 'counts': {'pink': 2, 'girl': 10}}))
            app = GeneratorApp(checkpoint, manifest, root / 'outputs', 'cpu')
            second_checkpoint = root / 'second.pt'
            second_payload = torch.load(checkpoint, weights_only=True)
            second_payload['step'] = 99
            torch.save(second_payload, second_checkpoint)
            second_manifest = root / 'second.json'
            second_manifest.write_text(json.dumps({'labels': ['pink'], 'counts': {'pink': 2}}))
            app.models['secondary'] = {'name': 'Second model', 'checkpoint': second_checkpoint,
                                       'manifest': second_manifest}
            class TestHandler(Handler):
                def log_message(self, *args):
                    pass
            TestHandler.app = app
            server = ThreadingHTTPServer(('127.0.0.1', 0), TestHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f'http://127.0.0.1:{server.server_port}'
            def get(path):
                with urlopen(url + path) as response:
                    return json.load(response)
            def post(payload):
                request = Request(url + '/api/generate', data=json.dumps(payload).encode(),
                                  headers={'Content-Type': 'application/json'})
                with urlopen(request) as response:
                    self.assertEqual(response.status, 202)
                    return json.load(response)
            try:
                self.assertEqual(get('/api/tags')['tags'][0]['tag'], 'girl')
                self.assertIn('secondary', [row['id'] for row in get('/api/models')['models']])
                self.assertEqual(get('/api/tags?model=secondary')['tags'], [{'tag': 'pink', 'count': 2}])
                with self.assertRaises(ValueError):
                    app.start({'tags': ['girl'], 'model': 'secondary'})
                with self.assertRaises(ValueError):
                    app.start({'tags': ['pink'], 'model': 'nonexistent'})
                with urlopen(Request(url + '/api/suggest', data=b'{"tags":["girl"]}',
                                     headers={'Content-Type': 'application/json'})) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual([row['tag'] for row in json.load(response)['tags']], ['pink'])
                with self.assertRaises(HTTPError) as invalid:
                    post({'tags': ['made up'], 'steps': 5})
                self.assertEqual(invalid.exception.code, 400)
                # A running job rejects competing generation rather than starting a second model.
                with patch('skin_generator_server.threading.Thread.start'):
                    held = app.start({'tags': ['pink'], 'steps': 5})
                with self.assertRaises(RuntimeError):
                    app.start({'tags': ['girl'], 'steps': 5})
                app.run(held['id'])
                job = get('/api/jobs/' + held['id'])
                self.assertEqual(job['state'], 'done', job.get('error'))
                self.assertEqual(len(job['results']), 4)
                seeds = {row['seed'] for row in job['results']}
                self.assertEqual(len(seeds), 4)
                for row in job['results']:
                    self.assertRegex(row['name'], r'^\d{8}_\d{6}_\d{6}_seed\d+\.png$')
                    with urlopen(url + row['url']) as response:
                        self.assertEqual(response.headers['Content-Type'], 'image/png')
                    with Image.open(app.output / row['name']) as png:
                        self.assertEqual(png.mode, 'RGBA')
                        self.assertEqual(png.size, (64, 64))
                    data = json.loads((app.output / row['name']).with_suffix('.json').read_text())
                    self.assertEqual(data['model'], 'slim')
                    self.assertEqual(str(data['seed']), row['seed'])
                self.assertEqual(len(get('/api/history')['results']), 4)
                skin_bytes = (app.output / job['results'][0]['name']).read_bytes()
                with urlopen(Request(url + '/api/describe-skin', data=skin_bytes,
                                     headers={'Content-Type': 'image/png'})) as response:
                    description = json.load(response)
                    self.assertGreater(len(description['tags']), 0)
                    self.assertLessEqual(len(description['tags']), 10)
                    for row in description['tags']:
                        self.assertTrue(0 <= row['score'] <= 1)
                        self.assertEqual(row['supported'], row['tag'] in app.allowed)
                for bad_image in (b'not a PNG', b''):
                    with self.assertRaises(HTTPError) as invalid_upload:
                        urlopen(Request(url + '/api/describe-skin', data=bad_image))
                    self.assertEqual(invalid_upload.exception.code, 400)
                import io
                wrong_size = io.BytesIO()
                Image.new('RGBA', (32, 32)).save(wrong_size, format='PNG')
                with self.assertRaises(ValueError):
                    app.describe_skin(wrong_size.getvalue())
                reloaded = GeneratorApp(checkpoint, manifest, app.output, 'cpu')
                self.assertEqual(len(reloaded.history()), 4)
                with self.assertRaises(HTTPError) as traversal:
                    urlopen(url + '/skins/%2e%2e/model.pt')
                self.assertEqual(traversal.exception.code, 403)
                # Also start an actual background request through HTTP.
                queued = post({'tags': ['pink'], 'steps': 5, 'model': 'secondary'})
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    job = get('/api/jobs/' + queued['id'])
                    if job['state'] != 'running':
                        break
                    time.sleep(.1)
                self.assertEqual(job['state'], 'done', job.get('error'))
                self.assertEqual(job['generator_model'], 'secondary')
                for row in job['results']:
                    metadata = json.loads((app.output / row['name']).with_suffix('.json').read_text())
                    self.assertEqual(metadata['generator_model'], 'secondary')
                    self.assertEqual(metadata['training_step'], 99)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
