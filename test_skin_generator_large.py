import unittest
import torch

from skin_generator import create_model, loss_batch, sample, schedule, transfer_generator
from skin_generator_large import LargeTagUNet, v_target, v_to_clean


class LargeGeneratorTests(unittest.TestCase):
    def test_finetune_remaps_old_tags_and_preserves_ema_weights(self):
        import copy
        source = create_model(2, 8, 'tag_ddpm_rgba_v2_attention')
        target = create_model(3, 8, 'tag_ddpm_rgba_v2_attention')
        ema = copy.deepcopy(target)
        new_embedding = target.tag_embedding.weight[3].detach().clone()
        checkpoint = {'labels': ['girl', 'pink'], 'model_state': source.state_dict(),
                      'ema_state': source.state_dict()}
        transfer_generator(target, ema, checkpoint, ['pink', 'girl', 'hoodie'])
        self.assertTrue(torch.equal(target.tag_embedding.weight[1], source.tag_embedding.weight[2]))
        self.assertTrue(torch.equal(target.tag_embedding.weight[2], source.tag_embedding.weight[1]))
        self.assertTrue(torch.equal(target.tag_embedding.weight[3], new_embedding))
        for key, value in source.state_dict().items():
            if key != 'tag_embedding.weight':
                self.assertTrue(torch.equal(target.state_dict()[key], value), key)
        for key, value in target.state_dict().items():
            self.assertTrue(torch.equal(ema.state_dict()[key], value), key)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_v_reconstructs_clean_even_at_near_zero_alpha(self):
        images, noise = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
        a = torch.tensor([1e-9, .3, .99])[:, None, None, None]
        noisy = a.sqrt() * images + (1 - a).sqrt() * noise
        recovered = v_to_clean(noisy, v_target(images, noise, a), a)
        self.assertTrue(torch.allclose(images, recovered, atol=1e-6))

    def test_large_model_learns_tags_and_handles_empty_condition(self):
        torch.manual_seed(42)
        model = LargeTagUNet(3, channels=8)
        images = torch.randn(2, 4, 64, 64).clamp(-1, 1)
        tags = torch.tensor([[1, 2], [0, 0]])
        loss = loss_batch(model, images, tags, schedule(20, 'cpu'), torch.ones(1, 1, 64, 64), 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.tag_embedding.weight.grad[1:].abs().sum()), 0)
        self.assertGreater(float(model.down_attention[1].cross.q_proj_weight.grad.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_v_sampling_is_finite_reproducible_and_conditioned(self):
        torch.manual_seed(11)
        model = create_model(3, 8, 'tag_ddpm_rgba_v2_attention').eval()
        first = sample(model, torch.tensor([[1, 2]]), 20, 5, 2.5, 42)
        second = sample(model, torch.tensor([[1, 2]]), 20, 5, 2.5, 42)
        other = sample(model, torch.tensor([[3, 0]]), 20, 5, 2.5, 42)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.allclose(first, other))
        self.assertTrue(torch.isfinite(first).all())
        self.assertEqual(first.shape, (1, 4, 64, 64))


if __name__ == '__main__':
    unittest.main()
