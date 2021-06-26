from random import randrange
import torch
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce
from functools import partial


'''
code based on repository: g-mlp-pytorch
'''


# functions
def exists(val):
    return val is not None


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val


def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers


# helper classes
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, memory=None):
        if memory != None:
            return self.fn(x, memory) + x
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, memory=None, **kwargs):
        x = self.norm(x)
        if memory != None:
            return self.fn(x, memory, **kwargs)
        return self.fn(x, **kwargs)


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


def MLP(num_patches, dim_ff):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
    return nn.Sequential(
        FeedForward(num_patches, dense=chan_first),
        FeedForward(dim_ff, dense=chan_last)
    )


class MLPEncoderLayer(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        dim,
        dim_ff,
    ):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )
        self.mlp = MLP(num_patches=num_patches, dim_ff=dim_ff)
        self.proj_out = nn.Linear(dim_ff, dim)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.mlp(x)
        x = self.proj_out(x)
        return x


class MLPDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        dim,
        dim_ff,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )
        self.mlp = MLP(num_patches=num_patches, dim_ff=dim_ff)
        self.proj_out = nn.Linear(dim_ff, dim)

    def forward(self, x, memory=None):
        res_1 = x
        x = self.proj_in(x)
        x = self.mlp(x)
        x = self.proj_out(x) + res_1

        x = x + memory
        x = self.norm(x)
        res_2 = x
        x = self.proj_in(x)
        x = self.mlp(x)
        x = self.proj_out(x) + res_2
        return x


# main classes
class MLPEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        dim,
        dim_ff,
        depth,
        training,
        prob_survival=1.
    ):
        super().__init__()
        self.prob_survival = prob_survival
        self.training = training
        self.layers = nn.ModuleList([Residual(PreNorm(dim, MLPEncoderLayer(num_patches=num_patches, dim=dim, dim_ff=dim_ff)))
                                     for i in range(depth)])

    def forward(self, x):
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        x = nn.Sequential(*layers)(x)
        return x


class MLPDecoder(nn.Module):
    def __init__(
            self,
            *,
            num_patches,
            dim,
            dim_ff,
            depth,
            training,
            prob_survival=1.
    ):
        super().__init__()
        self.prob_survival = prob_survival
        self.training = training
        self.depth = depth
        self.layer = Residual(PreNorm(dim, MLPDecoderLayer(num_patches=num_patches, dim=dim, dim_ff=dim_ff)))

    def forward(self, x, memory=None):
        layer = self.layer if not self.training else dropout_layers(self.layer, self.prob_survival)
        for i in range(self.depth):
            x = self.layer(x, memory=memory)
        return x


class EncoderDecoder(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            num_classes,
            dim,
            depth,
            num_queries,
            ff_mult=4,
            channels=3,
            prob_survival=1.,
            training=True
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert (image_height % patch_height) == 0 and (
                    image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, dim)
        self.training = training

        dim_ff = dim * ff_mult

        self.to_patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=patch_height, p2=patch_width),
            nn.Linear(channels * patch_height * patch_width, dim)
        )

        self.to_query_embed = nn.Sequential(
            Rearrange('n d -> d n'),
            nn.Linear(self.num_queries, patch_height * patch_width),
            Rearrange('d (b n) -> b n d', n=patch_height * patch_width)
        )

        self.prob_survival = prob_survival
        self.encoder = MLPEncoder(num_patches=num_patches, dim=dim, dim_ff=dim_ff, depth=depth, prob_survival=self.prob_survival, training = self.training)
        self.decoder = MLPDecoder(num_patches=num_patches, dim=dim, dim_ff=dim_ff, depth=depth, prob_survival=self.prob_survival, training = self.training)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embed(x)
        object_query = self.to_query_embed(self.query_embed.weight)
        x = self.encoder(x)
        x = self.decoder(object_query, memory=x)
        return self.to_logits(x)
