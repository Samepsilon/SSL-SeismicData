import numpy as np
import torch
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 sample_size=[3, 2000],
                 patch_size=10,
                 emb_dim=192,  # 192
                 num_layer=2,  # 12
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        n_channel, n_length = sample_size
        assert n_length % patch_size == 0, \
            f"n_length ({n_length}) must be divisible by patch_size ({patch_size})"
        num_patches = n_length // patch_size

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(num_patches, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv1d(n_channel, emb_dim, kernel_size=patch_size, stride=patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, x):
        # x: (batch, n_channel, n_length)
        patches = self.patchify(x)  # (batch, emb_dim, num_patches)
        patches = rearrange(patches, 'b c l -> l b c')  # (num_patches, batch, emb_dim)
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

    def forward_full(self, x):
        patches = self.patchify(x)  # (batch, emb_dim, num_patches)
        patches = rearrange(patches, 'b c l -> l b c')  # (num_patches, batch, emb_dim)
        patches = patches + self.pos_embedding

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features


class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 sample_size=[3, 2000],
                 patch_size=10,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        n_channel, n_length = sample_size
        assert n_length % patch_size == 0, \
            f"n_length ({n_length}) must be divisible by patch_size ({patch_size})"
        num_patches = n_length // patch_size

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(num_patches + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        """The output dimension of self.head is n_channel * patch_size (all channels for one time patch)"""
        self.head = torch.nn.Linear(emb_dim, n_channel * patch_size)
        self.patch2img = Rearrange('l b (c p) -> b c (l p)', c=n_channel, p=patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat(
            [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat(
            [features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)],
            dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        x = self.patch2img(patches)
        mask = self.patch2img(mask)

        return x, mask


class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 sample_shape=None,
                 patch_size=10,  # number of timesteps per patch
                 emb_dim=192,  # 192
                 encoder_layer=12,  # 12
                 encoder_head=8,  # 3
                 decoder_layer=6,
                 decoder_head=8,  # 3
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        if sample_shape is None:
            sample_shape = [3, 2000]
        self.encoder = MAE_Encoder(sample_shape, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(sample_shape, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, x):
        features, backward_indexes = self.encoder(x)
        predicted_x, mask = self.decoder(features, backward_indexes)
        return predicted_x, mask


class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes=2) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward_features(self, x):
        """Encoder forward pass, no masking. Returns the cls-token embedding
        (batch, emb_dim) — usable directly for clustering / embedding extraction,
        not just as an intermediate step before the classification head.
        """
        patches = self.patchify(x)
        patches = rearrange(patches, 'b c l -> l b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        return features[0]  # (batch, emb_dim)

    def forward(self, x):
        h = self.forward_features(x)
        logits = self.head(h)
        return logits
