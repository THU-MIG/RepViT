import torch
import time
from timm import create_model
import model
import utils
from fvcore.nn import FlopCountAnalysis

T0 = 5
T1 = 10

for n, batch_size, resolution in [
    ('repvit_m0_9', 1024, 224),
]:
    inputs = torch.randn(1, 3, resolution,
                            resolution)
    model = create_model(n, num_classes=1000)
    utils.replace_batchnorm(model)
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters / 1e6)
    flops = FlopCountAnalysis(model, inputs)
    print("flops: ", flops.total() / 1e9)