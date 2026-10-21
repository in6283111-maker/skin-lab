import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from next_tag import NextTagModel, exclude_given, make_batch, predict, read_data


class NextTagTests(unittest.TestCase):
    def test_tag_order_and_padding_do_not_change_prediction(self):
        torch.manual_seed(7)
        model = NextTagModel(5).eval()
        self.assertTrue(torch.allclose(model(torch.tensor([[1, 3]])),
                                       model(torch.tensor([[3, 1, 0, 0]]))))
        self.assertTrue(torch.isfinite(model(torch.zeros(1, 1, dtype=torch.long))).all())

    def test_hidden_target_never_occurs_in_context(self):
        import random
        rows = [(1, 2, 3)] * 32
        for augment in (False, True):
            tags, targets = make_batch(rows, random.Random(42), augment)
            self.assertFalse((tags == targets[:, None] + 1).any())
            masked = exclude_given(torch.randn(32, 3), tags)
            self.assertTrue(torch.isfinite(masked[torch.arange(32), targets]).all())

    def test_predict_excludes_given_tags_and_accepts_empty_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            model = NextTagModel(3)
            torch.save({'architecture': 'mean_tag_embedding_mlp_v1', 'labels': ['girl', 'pink', 'cute'],
                        'config': {'embedding_dim': 32, 'hidden_dim': 64}, 'state': model.state_dict()}, path)
            for names in ('Girl,Pink,girl', ''):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    predict(Namespace(model=path, tags=names, threads=2, top_k=10))
                results = json.loads(output.getvalue())
                if names:
                    self.assertEqual([row['tag'] for row in results], ['cute'])
                self.assertEqual(len({row['tag'] for row in results}), len(results))
                self.assertAlmostEqual(sum(row['score'] for row in results), 1, places=5)
            with self.assertRaises(ValueError):
                predict(Namespace(model=path, tags='unknown', threads=2, top_k=10))


if __name__ == '__main__':
    unittest.main()
