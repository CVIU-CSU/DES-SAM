_base_ = ['configs/_base_/datasets/coco_detection.py','configs/_base_/schedules/schedule_1x.py', 'configs/_base_/default_runtime.py']
# dataset settings
dataset_type = 'NucleiDataset'
data_root = 'data/CNSegTest/clusteredCellCOCO/'
# model settings
model = dict(
    type='FasterRCNNSimple',
    #pretrained=pretrained,
    data_preprocessor=dict(
        type='SAMInstDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        mask_stride=4,
        pairwise_size=3,
        pairwise_dilation=2,
        pairwise_color_thresh=0.3,
        bottom_pixels_removed=10),
    backbone=dict(
        type='ImageEncoderViT',
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2,5,8,11],
        window_size=14,
        out_chans=256,),
    neck=dict(
        type='SimpleFeaturePyramid',
        out_channels = 256,
        scale_factors = (4.0,2.0,1.0,0.5),
        top_block=None,
        square_pad=0,
        img_size = 1024,
        patch_size = 16,
        in_chans = 256,),
    rpn_head=dict(
        type='RPNHead',
        num_convs=2,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4,8, 16,32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIMaskHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16,32]),
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            #norm_cfg=head_norm_cfg,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0.,  0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            reg_decoded_bbox = True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        prompt_encoder = dict(
            type = 'PromptEncoder',
            embed_dim=256,
            image_embedding_size=(1024//16, 1024//16),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        ),
        mask_head=dict(
            type='MaskDecoder',
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            loss_mask=dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                naive_dice=True,
                eps=1.0,
                loss_weight=1.0),
            ),
        mask_label_head = dict(
            type='FrozenSAM',
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,),
        ori_level = 4
        ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.70),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=256,
            pos_weight=-1,
            mask_thr_binary=0.5,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.50),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.55,
            nms=dict(type='nms', iou_threshold=0.20),
            max_per_img=120,
            mask_thr_binary=0.5)))

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