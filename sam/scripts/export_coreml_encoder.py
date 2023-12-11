import torch

from timm import create_model

import torch
import torchvision
from argparse import ArgumentParser
from timm.models import create_model
import repvit_sam.modeling

parser = ArgumentParser()

parser.add_argument('--model', default='vit_t', type=str)
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--ckpt', default=None, type=str)
parser.add_argument('--samckpt', default=None, type=str)
parser.add_argument('--precision', default='fp16', type=str)

if __name__ == "__main__":
    # Load a pre-trained version of MobileNetV2
    args = parser.parse_args()
    model = create_model(args.model)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt)['model'])
    if args.samckpt:
        state = torch.load(args.samckpt, map_location='cpu')
        new_state = {}
        for k, v in state.items():
            if not 'image_encoder' in k:
                continue
            new_state[k.replace('image_encoder.', '')] = v
        model.load_state_dict(new_state)
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
        inputs=[ct.TensorType(shape=example_input.shape)],
        compute_precision=ct.precision.FLOAT16 if args.precision=='fp16' else ct.precision.FLOAT32
    )

    # Save the converted model.
    model.save(f"coreml/{args.model}_{resolution}.mlpackage")
