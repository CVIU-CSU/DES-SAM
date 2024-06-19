_base_ = ['configs/_base_/datasets/coco_detection.py','configs/_base_/schedules/schedule_1x.py', 'configs/_base_/default_runtime.py']
# dataset settings
dataset_type = 'NucleiDataset'
data_root = 'data/CNSegTest/clusteredCellCOCO/'

# model settings
model = dict(
    backbone=dict(
        depth=12,
        embed_dim=768,
        global_attn_indexes=[
            2,
            5,
            8,
            11,
        ],
        img_size=1024,
        mlp_ratio=4,
        num_heads=12,
        out_chans=256,
        patch_size=16,
        qkv_bias=True,
        type='ImageEncoderViT',
        use_rel_pos=True,
        window_size=14),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        bottom_pixels_removed=10,
        mask_stride=4,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        pairwise_color_thresh=0.3,
        pairwise_dilation=2,
        pairwise_size=3,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SAMInstDataPreprocessor'),
    neck=dict(
        img_size=1024,
        in_chans=256,
        out_channels=256,
        patch_size=16,
        scale_factors=(
            4.0,
            2.0,
            1.0,
            0.5,
        ),
        square_pad=0,
        top_block=None,
        type='SimpleFeaturePyramid'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            conv_out_channels=256,
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=0.1, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=True,
            reg_decoded_bbox=True,
            roi_feat_size=7,
            type='Shared4Conv1FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            loss_mask=dict(
                activate=True,
                eps=1.0,
                loss_weight=1.0,
                naive_dice=True,
                reduction='mean',
                type='DiceLoss',
                use_sigmoid=True),
            num_multimask_outputs=3,
            type='MaskDecoder'),
        mask_label_head=dict(
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_multimask_outputs=3,
            type='FrozenSAM'),
        ori_level=4,
        prompt_encoder=dict(
            embed_dim=256,
            image_embedding_size=(
                64,
                64,
            ),
            input_image_size=(
                1024,
                1024,
            ),
            mask_in_chans=16,
            type='PromptEncoder'),
        type='StandardRoIMaskHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        num_convs=2,
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=120,
            nms=dict(iou_threshold=0.2, type='nms'),
            score_thr=0.55),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.5, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=256,
            mask_thr_binary=0.5,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='FasterRCNNSimple')

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='ResizeLongestSide',keep_ratio=True, scale=1024),
    dict(type='Pad', pad_to_square=True),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='ResizeLongestSide',keep_ratio=True, scale=1024),
    dict(type='Pad', pad_to_square=True),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type='NucleiMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox'],
    classwise=True,
)
test_evaluator = dict(
    type='NucleiMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    classwise=True,
    metric=['bbox'],
    )
evaluation = dict(interval=20, metric=['acc', 'aji', 'pq'])
max_epochs = 50
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[30, 36],
        gamma=0.1)
]
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)