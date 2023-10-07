# Object Detection and Instance Segmentation

Detection and instance segmentation on MS COCO 2017 is implemented based on [MMDetection](https://github.com/open-mmlab/mmdetection).

## Models
| Model                   | $AP^b$ | $AP_{50}^b$ | $AP_{75}^b$ | $AP^m$ | $AP_{50}^m$ | $AP_{75}^m$ | Latency | Ckpt | Log |
|:---------------|:----:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| RepViT-M1.1 | 39.8  |  61.9   | 43.5  |    37.2    |  58.8      |  40.1        |     4.9ms    |   [M1.1](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_1_coco.pth)   | [M1.1](./logs/repvit_m1_1_coco.json) |
| RepViT-M1.5 | 41.6   | 63.2   | 45.3  | 38.6   | 60.5        | 41.5         |     6.4ms    |   [M1.5](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_5_coco.pth)   | [M1.5](./logs/repvit_m1_5_coco.json) |
| RepViT-M2.3 | 44.6   | 66.1        | 48.8        | 40.8   | 63.6        | 43.9  |     9.9ms    |   [M2.3](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m2_3_coco.pth)   | [M2.3](./logs/repvit_m2_3_coco.json) |

## Installation

Install [mmcv-full](https://github.com/open-mmlab/mmcv) and [MMDetection v2.28.2](https://github.com/open-mmlab/mmdetection/tree/v2.28.2),
Later versions should work as well. 
The easiest way is to install via [MIM](https://github.com/open-mmlab/mim)
```
pip install -U openmim
mim install mmcv-full==1.7.1
mim install mmdet==2.28.2
```

## Data preparation

Prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).
The dataset should be organized as 
```
detection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## Testing

We provide a multi-GPU testing script, specify config file, checkpoint, and number of GPUs to use: 
```
./dist_test.sh config_file path/to/checkpoint #GPUs --eval bbox segm
```

For example, to test RepViT-M1.1 on COCO 2017 on an 8-GPU machine, 

```
./dist_test.sh configs/mask_rcnn_repvit_m1_1_fpn_1x_coco.py path/to/repvit_m1_1_coco.pth 8 --eval bbox segm
```

## Training
Download ImageNet-1K pretrained weights into `./pretrain` 

We provide PyTorch distributed data parallel (DDP) training script `dist_train.sh`, for example, to train RepViT-M1.1 on an 8-GPU machine: 
```
./dist_train.sh configs/mask_rcnn_repvit_m1_1_fpn_1x_coco.py 8
```
Tips: specify configs and #GPUs!

