import torch

from timm import create_model
import model

import utils

import torch
import torchvision
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--model', default='repvit_m1_1', type=str)
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--ckpt', default=None, type=str)

if __name__ == "__main__":
    # Load a pre-trained version of MobileNetV2
    args = parser.parse_args()
    model = create_model(args.model, distillation=True)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt)['model'])
    utils.replace_batchnorm(model)
    model.eval()

    # Trace the model with random data.
    resolution = args.resolution
    example_input = torch.rand(1, 3, resolution, resolution) 
    traced_model = torch.jit.trace(model, example_input)
    out = traced_model(example_input)

    import coremltools as ct

    # Using image_input in the inputs parameter:
    # Convert to Core ML neural network using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(shape=example_input.shape)]
    )

    # Save the converted model.
    model.save(f"coreml/{args.model}_{resolution}.mlmodel")