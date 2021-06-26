import torch
from model.mlp import EncoderDecoder

model = EncoderDecoder(
    image_size = 256,
    patch_size = 16,
    dim = 512,
    depth = 12,
    num_classes = 1000,
    num_queries = 6
)

img = torch.randn(1, 3, 256, 256)
pred = model(img) # (1, 1000)
print(pred)