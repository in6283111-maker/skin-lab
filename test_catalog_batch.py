import json
import random
import unittest
from pathlib import Path

import torch
from torch import nn

from skin_generator import sample
from tools.catalog_prompts import EYES, HAIR, THEMES, conflicts, make_prompt


class CatalogBatchTests(unittest.TestCase):
    def test_conflicting_genders_and_color_aliases(self):
        for tags in (['boy', 'girls'], ['male', 'egirl'], ['green hair', 'brow hair'],
                     ['blonde', 'black hair'], ['green eyes', 'brown eyes'], ['summer', 'winter']):
            self.assertTrue(conflicts(tags), tags)
        for tags in (['boy', 'brown hair', 'green eyes', 'green hoodie'],
                     ['girl', 'blonde', 'blond hair'], ['boy', 'short hair', 'blue hair']):
            self.assertFalse(conflicts(tags), tags)

    def test_random_prompts_use_supported_compatible_tags(self):
        labels = set(HAIR + EYES + ['boy', 'girl'])
        labels.update(tag for theme in THEMES for group in theme for tag in group)
        rng = random.Random(73)
        unique = set()
        for _ in range(10000):
            tags = make_prompt(labels, rng)
            self.assertFalse(conflicts(tags))
            self.assertFalse(set(tags) - set(labels))
            self.assertEqual(len(set(tags) & {'boy', 'girl'}), 1)
            self.assertGreaterEqual(len(tags), 3)
            unique.add(tuple(tags))
        self.assertGreater(len(unique), 2000)

    def test_batch_uses_each_skin_seed_independently(self):
        class ZeroModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
            def forward(self, images, times, tags):
                return torch.zeros_like(images)
        model = ZeroModel()
        batched = sample(model, torch.tensor([[1], [2]]), 20, 4, 1, seeds=[17, 96])
        separate = torch.cat([sample(model, torch.tensor([[tag]]), 20, 4, 1, seed=seed)
                              for tag, seed in ((1, 17), (2, 96))])
        torch.testing.assert_close(batched, separate, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            sample(model, torch.tensor([[1], [2]]), 20, 4, seeds=[17])


if __name__ == '__main__':
    unittest.main()
