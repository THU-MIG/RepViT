_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
model = dict(
    backbone=dict(
        type='repvit_m1_5',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrain/repvit_m1_5_distill_300e.pth',
        ),
        out_indices=[4, 10, 36, 42]
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.05)  # 0.0001
optimizer_config = dict(grad_clip=None)
