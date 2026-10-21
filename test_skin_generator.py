"""Focused invariants: conditioning, diffusion gradients, tags and valid skin export."""
import tempfile
import unittest
import contextlib
import io
import json
from unittest.mock import patch
from pathlib import Path

import torch
from PIL import Image

from skin_generator import TagUNet, export_skin, loss_batch, parse_tags, schedule, uv_masks, parser, train, write_json, durable_checkpoint, load_training_checkpoint


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_tags_keep_multiword_labels_and_normalize(self):
        self.assertEqual(parse_tags(' Girl, Pink,Black Hair, girl '), ['black hair', 'girl', 'pink'])
        self.assertEqual(parse_tags('["Boy", "Blue"]'), ['blue', 'boy'])
        with self.assertRaises(ValueError):
            parse_tags('[1]')

    def test_corrupt_last_recovers_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            first = {'step': 1, 'epoch': 0, 'batch': 1, 'model_state': {},
                     'optimizer_state': {}, 'config': {}, 'labels': ['blue']}
            second = {**first, 'step': 2}
            durable_checkpoint(path, first)
            durable_checkpoint(path, second)
            # Replacement models a corrupted directory entry without modifying
            # the independent preceding checkpoint's hard-linked data.
            corrupt = path.with_name('broken.tmp')
            corrupt.write_bytes(b'incomplete archive')
            corrupt.replace(path)
            restored = load_training_checkpoint(path)
            self.assertEqual(restored['step'], 1)
            self.assertEqual(torch.load(path, weights_only=True)['step'], 1)
            self.assertEqual(len(list(path.parent.glob('last.corrupt.*.pt'))), 1)

    def test_corrupt_last_recovers_best_without_resetting_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            path.write_bytes(b'partial write')
            payload = {'step': 100, 'epoch': 1, 'batch': 0, 'model_state': {},
                       'optimizer_state': {}, 'config': {}, 'labels': ['blue']}
            durable_checkpoint(path.with_name('best.pt'), payload)
            self.assertEqual(load_training_checkpoint(path)['step'], 100)

    def test_status_write_retries_temporary_windows_file_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'status.json'
            original = Path.replace
            attempts = []

            def temporarily_locked(source, target):
                attempts.append(source)
                if len(attempts) == 1:
                    raise PermissionError('simulated Windows reader lock')
                return original(source, target)

            with patch.object(Path, 'replace', temporarily_locked):
                write_json(destination, {'step': 123})
            self.assertEqual(json.loads(destination.read_text())['step'], 123)
            self.assertEqual(len(attempts), 2)

    def test_diffusion_updates_weights_and_uses_condition(self):
        torch.manual_seed(7)
        model = TagUNet(3, channels=8)
        images = torch.randn(2, 4, 64, 64).clamp(-1, 1)
        tags = torch.tensor([[1, 2], [3, 0]])
        alpha = schedule(100, torch.device('cpu'))
        self.assertTrue(torch.all(alpha[1:] < alpha[:-1]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.input.weight.detach().clone()
        loss = loss_batch(model, images, tags, alpha, torch.ones(1, 1, 64, 64), 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.tag_embedding.weight.grad[1:].abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.input.weight))
        with torch.no_grad():
            times = torch.tensor([10, 10])
            self.assertFalse(torch.allclose(model(images, times, tags), model(images, times, torch.zeros_like(tags))))

    def test_export_is_rgba_with_opaque_base_and_empty_unused_uv(self):
        pixels = torch.zeros(4, 64, 64)
        pixels[:3] = .25
        pixels[3] = .5
        base, valid = uv_masks()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'skin.png'
            export_skin(pixels, path)
            with Image.open(path) as image:
                self.assertEqual(image.size, (64, 64))
                self.assertEqual(image.mode, 'RGBA')
                rgba = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(64, 64, 4)
            self.assertTrue(torch.all(rgba[:, :, 3][base] == 255))
            self.assertTrue(torch.all(rgba[~valid] == 0))
            self.assertTrue(torch.all(rgba[:, :, 0][base] == 128))

    def test_cpu_resume_matches_uninterrupted_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            (data / 'skins').mkdir(parents=True)
            rows = []
            for index in range(8):
                name = f'{index}.png'
                Image.new('RGBA', (64, 64), (index * 30, 64, 128, 255)).save(data / 'skins' / name)
                rows.append({'filename': name, 'tags': ['Blue', 'Boy' if index % 2 else 'Girl']})
            (data / 'data.json').write_text(json.dumps(rows), encoding='utf-8')
            options = ['train', '--data', str(data), '--min-tag-count', '1', '--channels', '8',
                       '--batch-size', '2', '--workers', '0', '--diffusion-steps', '20',
                       '--device', 'cpu', '--cpu-threads', '2', '--log-every', '1']
            with contextlib.redirect_stdout(io.StringIO()):
                train(parser().parse_args(options + ['--output', str(root / 'full'), '--max-steps', '3']))
                train(parser().parse_args(options + ['--output', str(root / 'resumed'), '--max-steps', '1']))
                train(parser().parse_args(options + ['--output', str(root / 'resumed'), '--max-steps', '3']))
            full = torch.load(root / 'full' / 'last.pt', weights_only=True)
            resumed = torch.load(root / 'resumed' / 'last.pt', weights_only=True)
            self.assertEqual(full['step'], 3)
            self.assertEqual(resumed['step'], 3)
            for name, value in full['model_state'].items():
                self.assertTrue(torch.equal(value, resumed['model_state'][name]), name)


if __name__ == '__main__':
    unittest.main()
