import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from skin_generator import (ARCHITECTURE, create_model, export_skin, generate,
                            load_edit_skin, parser, sample, uv_masks, ROOT)


class SkinEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_edit_sampling_boundaries_and_conditioning(self):
        source = torch.zeros(1, 4, 64, 64)
        for architecture in (ARCHITECTURE, 'tag_ddpm_rgba_v2_attention'):
            with self.subTest(architecture=architecture):
                torch.manual_seed(10)
                model = create_model(2, 8, architecture)
                tags = torch.tensor([[1]])
                self.assertTrue(torch.equal(sample(model, tags, 20, 5, initial_image=source, strength=0),
                                            (source + 1) / 2))
                self.assertTrue(torch.equal(sample(model, tags, 20, 5),
                                            sample(model, tags, 20, 5, initial_image=source, strength=1)))
                edited = sample(model, tags, 20, 5, initial_image=source, strength=.5)
                repeated = sample(model, tags, 20, 5, initial_image=source, strength=.5)
                changed_tags = sample(model, torch.tensor([[2]]), 20, 5, initial_image=source, strength=.5)
                changed_source = sample(model, tags, 20, 5, initial_image=source + .8, strength=.5)
                self.assertTrue(torch.isfinite(edited).all())
                self.assertTrue(torch.equal(edited, repeated))
                self.assertFalse(torch.allclose(edited, changed_tags))
                self.assertFalse(torch.allclose(edited, changed_source))
                for invalid in (-.1, 1.1, float('nan')):
                    with self.assertRaises(ValueError):
                        sample(model, tags, 20, 5, initial_image=source, strength=invalid)

    def test_cli_edit_and_original_generate_export(self):
        self.assertIsNone(parser().parse_args(['generate', '--tags', 'Pink']).seed)
        self.assertEqual(parser().parse_args(['generate', '--tags', 'Pink', '--seed', '42']).seed, 42)
        self.assertEqual(parser().parse_args(['generate', '--tags', 'Pink']).output, ROOT / 'generated_skins')
        self.assertEqual(parser().parse_args(['edit', '--tags', 'Pink', '--input', 'skin.png']).output,
                         ROOT / 'edited_skins')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = torch.rand(4, 64, 64)
            pixels[3] = uv_masks()[1].float()
            source = root / 'source.png'
            export_skin(pixels, source)
            model = create_model(1, 8, ARCHITECTURE)
            checkpoint = root / 'test.pt'
            torch.save(dict(architecture=ARCHITECTURE, labels=['pink'], step=7,
                            config=dict(channels=8, diffusion_steps=20),
                            ema_state=model.state_dict(), model_state=model.state_dict()), checkpoint)
            common = ['--tags', 'Pink', '--count', '1', '--device', 'cpu',
                      '--checkpoint', str(checkpoint), '--output', str(root / 'outputs'),
                      '--sampling-steps', '5']
            for command, extra in [('generate', []), ('edit', ['--input', str(source), '--strength', '0']),
                                   ('edit', ['--input', str(source), '--strength', '.5', '--seed', '43'])]:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    generate(parser().parse_args([command, *common, *extra]))
                result = Path(json.loads(output.getvalue())[0])
                metadata = json.loads(result.with_suffix('.json').read_text())
                self.assertIsInstance(metadata['seed'], int)
                self.assertIn(f"seed{metadata['seed']}", result.stem)
                if '--seed' in extra:
                    self.assertEqual(metadata['seed'], 43)
                with Image.open(result) as image:
                    self.assertEqual(image.size, (64, 64))
                    self.assertEqual(image.mode, 'RGBA')
                    if command == 'edit' and extra[-1] == '0':
                        with Image.open(source) as original:
                            self.assertEqual(image.tobytes(), original.tobytes())
                if command == 'edit':
                    metadata = json.loads(result.with_suffix('.json').read_text())
                    self.assertEqual(metadata['input_skin'], str(source.resolve()))
            bad = root / 'bad.png'
            Image.new('RGBA', (64, 32)).save(bad)
            with self.assertRaises(ValueError):
                load_edit_skin(bad)


if __name__ == '__main__':
    unittest.main()
