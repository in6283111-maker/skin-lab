"""Larger tag-attentive pixel U-Net; v-prediction avoids division by tiny alpha."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from skin_generator import ResBlock, uv_masks

ARCHITECTURE_LARGE = 'tag_ddpm_rgba_v2_attention'


class SpatialTagAttention(nn.Module):
    def __init__(self, channels, tag_dim, self_attention=False):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.cross = nn.MultiheadAttention(channels, 8, kdim=tag_dim, vdim=tag_dim, batch_first=True)
        self.self_attention = nn.MultiheadAttention(channels, 8, batch_first=True) if self_attention else None
        self.self_norm = nn.LayerNorm(channels) if self_attention else None
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x, tokens, padding):
        b, c, h, w = x.shape
        queries = self.norm(x).flatten(2).transpose(1, 2)
        update, _ = self.cross(queries, tokens, tokens, key_padding_mask=padding, need_weights=False)
        if self.self_attention is not None:
            normalized = self.self_norm(queries + update)
            spatial, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
            update = update + spatial
        return x + self.output(update.transpose(1, 2).reshape(b, c, h, w))


class LargeTagUNet(nn.Module):
    prediction_type = 'v'

    def __init__(self, num_tags, channels=96, gradient_checkpointing=True):
        super().__init__()
        dim = channels * 4
        self.time_dim = dim
        self.gradient_checkpointing = gradient_checkpointing
        self.tag_embedding = nn.Embedding(num_tags + 1, dim, padding_idx=0)
        self.null_token = nn.Parameter(torch.randn(1, 1, dim) * .02)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.tag_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        base, valid = uv_masks()
        coordinate = torch.linspace(-1, 1, 64)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing='ij')
        self.register_buffer('uv_features', torch.stack([xx, yy, base.float(), valid.float()]))
        self.input = nn.Conv2d(8, channels, 3, padding=1)
        widths = [channels, channels * 2, channels * 4, channels * 4]
        self.down_blocks = nn.ModuleList([nn.ModuleList([ResBlock(w, w, dim), ResBlock(w, w, dim)]) for w in widths])
        self.down_attention = nn.ModuleList([nn.Identity()] + [SpatialTagAttention(w, dim, i >= 2) for i, w in enumerate(widths) if i > 0])
        self.downsamples = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(widths, widths[1:])])
        self.middle = nn.ModuleList([ResBlock(widths[-1], widths[-1], dim), ResBlock(widths[-1], widths[-1], dim)])
        self.middle_attention = SpatialTagAttention(widths[-1], dim, True)
        self.up_blocks = nn.ModuleList([nn.ModuleList([ResBlock(a + b, b, dim), ResBlock(b, b, dim)])
                                       for a, b in zip(reversed(widths[1:]), reversed(widths[:-1]))])
        self.up_attention = nn.ModuleList([SpatialTagAttention(widths[2], dim, True),
                                          SpatialTagAttention(widths[1], dim, False), nn.Identity()])
        self.output = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, 4, 3, padding=1))

    def call(self, module, *args):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x, timesteps, tags):
        half = self.time_dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / (half - 1))
        angles = timesteps.float()[:, None] * frequencies[None]
        time_condition = self.time_mlp(torch.cat([angles.sin(), angles.cos()], 1))
        embeddings = self.tag_embedding(tags)
        present = (tags != 0).unsqueeze(-1)
        pooled = (embeddings * present).sum(1) / present.sum(1).clamp(min=1)
        pooled = torch.where(present.any(1), pooled, self.null_token[:, 0])
        condition = time_condition + self.tag_mlp(pooled)
        tokens = torch.cat([self.null_token.expand(len(tags), -1, -1), embeddings], 1)
        padding = torch.cat([torch.zeros(len(tags), 1, dtype=torch.bool, device=x.device), tags == 0], 1)
        h = self.input(torch.cat([x, self.uv_features[None].expand(len(x), -1, -1, -1)], 1))
        skips = []
        for index, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = self.call(block, h, condition)
            if index > 0:
                h = self.call(self.down_attention[index], h, tokens, padding)
            if index < len(self.downsamples):
                skips.append(h)
                h = self.downsamples[index](h)
        for block in self.middle:
            h = self.call(block, h, condition)
        h = self.call(self.middle_attention, h, tokens, padding)
        for index, (blocks, skip) in enumerate(zip(self.up_blocks, reversed(skips))):
            h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
            h = torch.cat([h, skip], 1)
            for block in blocks:
                h = self.call(block, h, condition)
            if index < 2:
                h = self.call(self.up_attention[index], h, tokens, padding)
        return self.output(h)


def v_target(images, noise, alpha):
    return alpha.sqrt() * noise - (1 - alpha).sqrt() * images


def v_to_clean(noisy, prediction, alpha):
    return alpha.sqrt() * noisy - (1 - alpha).sqrt() * prediction


def v_loss_batch(model, images, tags, alpha, weight, drop_probability=.1, generator=None):
    times = torch.randint(len(alpha), (len(images),), device=images.device, generator=generator)
    noise = torch.randn(images.shape, device=images.device, generator=generator)
    a = alpha[times, None, None, None]
    noisy = a.sqrt() * images + (1 - a).sqrt() * noise
    if drop_probability:
        tags = tags.clone()
        tags[torch.rand(len(tags), device=tags.device) < drop_probability] = 0
    prediction = model(noisy, times, tags).float()
    target = v_target(images, noise, a)
    per_example = ((prediction - target).square() * weight).sum((1, 2, 3)) / (4 * weight.sum())
    snr = a.flatten() / (1 - a.flatten()).clamp(min=1e-8)
    min_snr_weight = snr.clamp(max=5) / (snr + 1)
    loss = (per_example * min_snr_weight).mean()
    # Low-noise alpha/detail reconstruction discourages losing sharp texel boundaries.
    clean = v_to_clean(noisy, prediction, a)
    detail = (a.flatten() > .5).float()
    alpha_loss = ((clean[:, 3:4] - images[:, 3:4]).abs() * weight).sum((1, 2, 3)) / weight.sum()
    mask = uv_masks()[1].to(images.device)
    edge_losses = []
    for axis in (2, 3):
        predicted_edge = torch.diff(clean[:, :3], dim=axis)
        true_edge = torch.diff(images[:, :3], dim=axis)
        edge_mask = (mask[1:, :] & mask[:-1, :]) if axis == 2 else (mask[:, 1:] & mask[:, :-1])
        edge_losses.append(((predicted_edge - true_edge).abs() * edge_mask).sum((1, 2, 3)) / (3 * edge_mask.sum()))
    loss += .05 * ((alpha_loss + sum(edge_losses) / 2) * detail).mean()
    return loss
