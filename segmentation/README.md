# Semantic Segmentation 

Segmentation on ADE20K is implemented based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

## Models
| Model | mIoU | Latency | Ckpt | Log |
|:---------------|:----:|:---:|:--:|:--:|
| RepViT-M1.1 |   40.6   |     4.9ms    |   [M1.1](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_1_ade20k.pth)   | [M1.1](./logs/repvit_m1_1_ade20k.json) |
| RepViT-M1.5 |   43.6   |     6.4ms    |   [M1.5](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_5_ade20k.pth)   | [M1.5](./logs/repvit_m1_5_ade20k.json) |
| RepViT-M2.3 |   46.1   |     9.9ms    |   [M2.3](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m2_3_ade20k.pth)   | [M2.3](./logs/repvit_m2_3_ade20k.json) |

The backbone latency is measured with image crops of 512x512 on iPhone 12 by Core ML Tools.

## Requirements
Install [mmcv-full](https://github.com/open-mmlab/mmcv) and [MMSegmentation v0.30.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0). 
Later versions should work as well. 
The easiest way is to install via [MIM](https://github.com/open-mmlab/mim)
```
pip install -U openmim
mim install mmcv-full==1.7.1
mim install mmseg==0.30.0
```

## Data preparation

We benchmark RepViT on the challenging ADE20K dataset, which can be downloaded and prepared following [insructions in MMSeg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets). 
The data should appear as: 
```
├── segmentation
│   ├── data
│   │   ├── ade
│   │   │   ├── ADEChallengeData2016
│   │   │   │   ├── annotations
│   │   │   │   │   ├── training
│   │   │   │   │   ├── validation
│   │   │   │   ├── images
│   │   │   │   │   ├── training
│   │   │   │   │   ├── validation

```



## Testing

We provide a multi-GPU testing script, specify config file, checkpoint, and number of GPUs to use: 
```
./tools/dist_test.sh config_file path/to/checkpoint #GPUs --eval mIoU
```

For example, to test RepViT-M1.1 on ADE20K on an 8-GPU machine, 

```
./tools/dist_test.sh configs/sem_fpn/fpn_repvit_m1_1_ade20k_40k.py path/to/repvit_m1_1_ade20k.pth 8 --eval mIoU
```

## Training 
Download ImageNet-1K pretrained weights into `./pretrain` 

We provide PyTorch distributed data parallel (DDP) training script `dist_train.sh`, for example, to train RepViT-M1.1 on an 8-GPU machine: 
```
./tools/dist_train.sh configs/sem_fpn/fpn_repvit_m1_1_ade20k_40k.py 8
```
Tips: specify configs and #GPUs!
