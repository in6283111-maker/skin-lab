import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from download_models import download, safe_target
from launch import choose_backend


class DistributionTests(unittest.TestCase):
    def test_download_checksums_and_protects_changed_local_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, destination = root / 'cache', root / 'app'
            cache.mkdir()
            destination.mkdir()
            data = cache / 'weights.pt'
            data.write_bytes(b'public weights')
            filename = 'checkpoints/generator_large/last.pt'
            index = cache / 'models.json'
            index.write_text(json.dumps({'models': {'generator_large': {'files': [
                {'path': filename, 'sha256': hashlib.sha256(data.read_bytes()).hexdigest(), 'size': data.stat().st_size}]}}}))
            def fetch(repo, name, revision):
                self.assertEqual(revision, 'fixed-commit')
                return str(index if name == 'models.json' else data)
            with patch('huggingface_hub.HfApi') as api, patch('huggingface_hub.hf_hub_download', side_effect=fetch):
                api.return_value.model_info.return_value = SimpleNamespace(sha='fixed-commit')
                download('owner/models', 'main', destination)
                target = destination / filename
                self.assertEqual(target.read_bytes(), data.read_bytes())
                target.write_bytes(b'my fine-tuned model')
                with self.assertRaises(ValueError):
                    download('owner/models', 'main', destination)
                self.assertEqual(target.read_bytes(), b'my fine-tuned model')
                download('owner/models', 'main', destination, force=True)
                self.assertEqual(target.read_bytes(), data.read_bytes())
                target.unlink()
                data.write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    download('owner/models', 'main', destination)
                self.assertFalse(target.exists())

    def test_model_paths_cannot_escape_checkpoint_directory(self):
        root = Path.cwd()
        for name in ('../secret', '/secret', 'checkpoints/../../secret', 'checkpoints/../.env'):
            with self.assertRaises(ValueError):
                safe_target(root, name)
        self.assertEqual(safe_target(root, 'checkpoints/next_tag/model.pt'), root / 'checkpoints/next_tag/model.pt')

    def test_auto_backend_handles_macos_and_missing_gpu(self):
        with patch('launch.platform.system', return_value='Darwin'):
            self.assertEqual(choose_backend('auto'), 'cpu')
        with patch('launch.platform.system', return_value='Windows'), patch('launch.shutil.which', return_value=None):
            self.assertEqual(choose_backend('auto'), 'cpu')


if __name__ == '__main__':
    unittest.main()
