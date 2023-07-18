# [RepViT: Revisiting  Mobile CNN From ViT Perspective](https://arxiv.org/abs/2307.09283)

Official PyTorch implementation of **RepViT**, from the following paper:

[RepViT: Revisiting  Mobile CNN From ViT Perspective](https://arxiv.org/abs/2307.09283).\
Ao Wang, Hui Chen, Zijia Lin, Hengjun Pu, and Guiguang Ding\
[[`arXiv`](https://arxiv.org/abs/2307.09283)]

<p align="center">
  <img src="figures/latency.png" width=70%> <br>
  Models are trained on ImageNet-1K and deployed on iPhone 12 with Core ML Tools to get latency.
</p>

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Recently, lightweight Vision Transformers (ViTs) demonstrate superior performance and lower latency compared with lightweight Convolutional Neural Networks (CNNs) on resource-constrained mobile devices. This improvement is usually attributed to the multi-head self-attention module, which enables the model to learn global representations. However, the architectural disparities between lightweight ViTs and lightweight CNNs have not been adequately examined. In this study, we revisit the efficient design of lightweight CNNs and emphasize their potential for mobile devices. We incrementally enhance the mobile-friendliness of a standard lightweight CNN, specifically MobileNetV3, by integrating the efficient architectural choices of lightweight ViTs. To this end, we present a new family of pure lightweight CNNs, namely RepViT. Extensive experiments show that RepViT outperforms existing state-of-the-art lightweight ViTs and exhibits favorable latency in various vision tasks. On ImageNet, RepViT achieves over 80\% top-1 accuracy with nearly 1ms latency on an iPhone 12, which is the first time for a lightweight model, to the best of our knowledge. Our largest model, RepViT-M3, obtains 81.4\% accuracy with only 1.3ms latency.
</details>

<br>

## Classification on ImageNet-1K

### Models

| Model | Top-1 (300)| #params | MACs | Latency | Ckpt | Core ML | Log |
|:---------------|:----:|:---:|:--:|:--:|:--:|:--:|:--:|
| RepViT-M1 |   78.5   |     5.1M    |   0.8G   |      0.9ms     |  [M1](https://github.com/jameslahm/RepViT/releases/download/untagged-75eb9e1fea235b938f50/repvit_m1_distill_300.pth)    |   [M1](https://github.com/jameslahm/RepViT/releases/download/untagged-75eb9e1fea235b938f50/repvit_m1_224.mlmodel)  | [M1](./logs/repvit_m1_train.log) |
| RepViT-M2 |   80.6   |     8.8M    |   1.4G   |      1.1ms     |  [M2](https://github.com/jameslahm/RepViT/releases/download/untagged-75eb9e1fea235b938f50/repvit_m2_distill_300.pth)    |   [M2](https://github.com/jameslahm/RepViT/releases/download/untagged-75eb9e1fea235b938f50/repvit_m2_224.mlmodel)  | [M2](./logs/repvit_m2_train.log) |
| RepViT-M3 |   81.4   |     10.1M    |   1.9G   |      1.3ms     |  [M3](https://github.com/jameslahm/RepViT/releases/download/untagged-75eb9e1fea235b938f50/repvit_m3_distill_300.pth)    |   [M3](https://github.com/jameslahm/RepViT/releases/download/untagged-75eb9e1fea235b938f50/repvit_m3_224.mlmodel)  | [M3](./logs/repvit_m3_train.log) |

Tips: Convert a training-time RepViT into the inference-time structure
```
from timm.models import create_model
import utils

model = create_model('repvit_m1')
utils.replace_batchnorm(model)
```

## Latency Measurement 

The latency reported in RepViT for iPhone 12 (iOS 16) uses the benchmark tool from [XCode 14](https://developer.apple.com/videos/play/wwdc2022/10027/).
For example, here is a latency measurement of RepViT-M1:

![](./figures/repvit_m1_latency.png)

Tips: export the model to Core ML model
```
python export_coreml.py --model repvit_m1 --ckpt pretrain/repvit_m1_distill_300.pth
```
Tips: measure the throughput on GPU
```
python speed_gpu.py --model repvit_m1
```


## ImageNet  

### Prerequisites
`conda` virtual environment is recommended. 
```
conda create -n repvit python=3.8
pip install -r requirements.txt
```

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/. The training and validation data are expected to be in the `train` folder and `val` folder respectively:
```
|-- /path/to/imagenet/
    |-- train
    |-- val
```

### Training
To train RepViT-M1 on an 8-GPU machine:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12346 --use_env main.py --model repvit_m1 --data-path ~/imagenet --dist-eval
```
Tips: specify your data path and model name! 

### Testing 
For example, to test RepViT-M1:
```
python main.py --eval --model repvit_m3 --resume pretrain/repvit_m3_distill_300.pth --data-path ~/imagenet
```

## Downstream Tasks
[Object Detection and Instance Segmentation](detection/README.md)<br>
[Semantic Segmentation](segmentation/README.md)

## Acknowledgement

Classification (ImageNet) code base is partly built with [LeViT](https://github.com/facebookresearch/LeViT), [PoolFormer](https://github.com/sail-sg/poolformer) and [EfficientFormer](https://github.com/snap-research/EfficientFormer). 

The detection and segmentation pipeline is from [MMCV](https://github.com/open-mmlab/mmcv) ([MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)). 

Thanks for the great implementations! 

## Citation

If our code or models help your work, please cite our papers:
```BibTeX

```
