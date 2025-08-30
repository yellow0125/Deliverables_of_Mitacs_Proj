_base_ = 'mmdet::common/ssj_scp_270k_coco-instance.py'

custom_imports = dict(
    imports=['projects.CO-DETR.codetr'], allow_failed_imports=False)

# model settings
num_dec_layer = 6
loss_lambda = 2.0
num_classes = 4

image_size = (1024, 682)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=False)
]

data_root = '/scratch/yang.zhiyu/tif_test/log_data/'
metainfo = {
    'classes': ("Alive Tree", "Debris", "Dead Tree", "Beetle Fire Tree"),
    'palette': [
        (70, 230, 70),
        (255, 250, 0),
        (255, 69, 0),
        (255, 5, 5)
    ]
}

model = dict(
    type='CoDETR',
    # If using the lsj augmentation,
    # it is recommended to set it to True.
    use_lsj=True,
    # detr: 52.1
    # one-stage: 49.4
    # two-stage: 47.9
    eval_module='detr',  # in ['detr', 'one-stage', 'two-stage']
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[1.1473892, 1.1473892, 1.1473892],
        std=[3.029705, 3.029705, 3.029705],
        bgr_to_rgb=False,
        pad_mask=False,
        batch_augments=batch_augments),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    query_head=dict(
        type='CoDINOHead',
        num_query=900,
        num_classes=num_classes,
        in_channels=2048,
        as_two_stage=True,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=1.0,
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type='CoDinoTransformer',
            with_coord_feat=False,
            num_co_heads=2,  # ATSS Aux Head + Faster RCNN Aux Head
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                # number of layers that use checkpoint.
                # The maximum value for the setting is num_layers.
                # FairScale must be installed for it to work.
                with_cp=4,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=5,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(  # Different from the DINO
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0 * num_dec_layer * loss_lambda),
        loss_bbox=dict(
            type='L1Loss', loss_weight=1.0 * num_dec_layer * loss_lambda)),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ])),
        dict(
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
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
    ],
    test_cfg=[
        # Deferent from the DINO, we use the NMS.
        dict(
            max_per_img=300,
            # NMS can improve the mAP by 0.2.
            nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)),
        dict(
            # atss bbox head:
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ])

# # # LSJ + CopyPaste
# # load_pipeline = [
# #     dict(type='LoadImageFromFile'),
# #     dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
# #     dict(
# #         type='RandomResize',
# #         scale=image_size,
# #         ratio_range=(0.1, 2.0),
# #         keep_ratio=True),
# #     dict(
# #         type='RandomCrop',
# #         crop_type='absolute_range',
# #         crop_size=image_size,
# #         recompute_bbox=True,
# #         allow_negative_crop=True),
# #     dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
# #     dict(type='RandomFlip', prob=0.5),
# #     dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
# # ]

# # train_pipeline = [
# #     dict(type='CopyPaste', max_num_pasted=100),
# #     dict(type='PackDetInputs')
# # ]

# # train_dataloader = dict(
# #     sampler=dict(type='DefaultSampler', shuffle=True),
# #     dataset=dict(
# #         pipeline=train_pipeline,
# #         dataset=dict(
# #             filter_cfg=dict(filter_empty_gt=False), pipeline=load_pipeline)))

# # # follow ViTDet
# # test_pipeline = [
# #     dict(type='LoadImageFromFile'),
# #     dict(type='Resize', scale=image_size, keep_ratio=True),  # diff
# #     dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
# #     dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
# #     dict(
# #         type='PackDetInputs',
# #         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
# #                    'scale_factor'))
# # ]

# # val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
# # test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.0001, capturable = True),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

# # val_evaluator = dict(metric='bbox')
# # test_evaluator = val_evaluator

max_epochs = 300
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# default_hooks = dict(
#     checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=3))
# log_processor = dict(by_epoch=True)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)











# # # The new config inherits a base config to highlight the necessary modification
# # _base_ = './mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py'

# # # We also need to change the num_classes in head to match the dataset's annotation
# # num_classes = 4
# # image_size = (1024, 682)
# # # dataset_type = 'CocoDataset'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
# # data_root = '/scratch/yang.zhiyu/tif_test/log_data/'
# # metainfo = {
# #     'classes': ("Alive Tree", "Debris", "Dead Tree", "Beetle Fire Tree"),
# #     'palette': [
# #         (70, 230, 70),
# #         (255, 250, 0),
# #         (255, 69, 0),
# #         (255, 5, 5)
# #     ]
# # }
# # batch_augments = [
# #     dict(type='BatchFixedSizePad', size=image_size, pad_mask=False)
# # ]
# model = dict(
#     data_preprocessor=dict(
#         type='DetDataPreprocessor',
#         batch_augments = batch_augments,
#         mean=[1.1473892, 1.1473892, 1.1473892],
#         std=[3.029705, 3.029705, 3.029705]),
#     backbone=dict(
#         _delete_=True,
#         type='SwinTransformer',
#         pretrain_img_size=384,
#         embed_dims=192,
#         depths=[2, 2, 18, 2],
#         num_heads=[6, 12, 24, 48],
#         window_size=12,
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.3,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         # Please only add indices that would be used
#         # in FPN, otherwise some parameter will not be used
#         with_cp=False,
#         convert_weights=True,
#         init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
#     neck=dict(in_channels=[192, 384, 768, 1536]),
#     query_head=dict(transformer=dict(encoder=dict(with_cp=6)))
# )

# # resume = True
# max_epochs = 300 # 36 default

# # samples_per_gpu=1

# # optimizer
# # optim_wrapper = dict(
# #     optimizer=dict(
# #         capturable = True
# #     )
# # )
# # optimizer=dict(capturable = True)

# # Modify dataset related setting
load_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile', to_float32=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]

train_dataloader = dict(
    # dataset=dict(
        dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        pipeline=load_pipeline,
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        pipeline=load_pipeline,
        data_prefix=dict(img='valid/')))
test_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/_annotations.coco.json',
        pipeline=load_pipeline,
        data_prefix=dict(img='test/')))

# Modify metric related settings
val_evaluator = dict(
    # type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    # metric='bbox',
    classwise=True,  # Enable per-class evaluation
)

test_evaluator = dict(
    # type='CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    # metric='bbox',
    classwise=True,  # Enable per-class evaluation
)

# Modify evaluation settings
evaluation = dict(
    # metric=['bbox'],
    classwise=True,  # Enable per-class evaluation
    interval=1)  # Evaluate every epoch

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=2,  # only keep latest 2 checkpoints
        save_best='auto'
    ),
    # early_stopping = dict(
    #     type="EarlyStoppingHook",
    #     patience=10,
    #     monitor="coco/bbox_mAP",
    #     min_delta=0.005)
)

# train_cfg = dict(
#     type='EpochBasedTrainLoop',
#     max_epochs=300,  # Override the base config's epochs
#     val_interval=1
# )
# runner = dict(type='EpochBasedRunner', max_epochs=300)

work_dir="/scratch/yang.zhiyu/tif_test/ff_codet_log_1227"

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_swinl_30e.pth'






# _base_ = './mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py'

# data_root = '/scratch/yang.zhiyu/tif_test/log_data/' # dataset root

# train_batch_size_per_gpu = 2
# train_num_workers = 2

# max_epochs = 150
# stage2_num_epochs = 2
# base_lr = 0.00008


# metainfo = {
#     'classes': ('Alive Tree','Beetle Fire Tree','Dead Tree','Debris',),
#      'palette': [
#         (70, 230, 70),
#         (255, 250, 0),
#         (255, 69, 0),
#         (255, 5, 5)
#     ]
# }

# train_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         pipeline=[
#             dict(type='LoadImageFromFile', imdecode_backend='tifffile', to_float32=True),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(type='Resize',scale_factor=1.0, keep_ratio=True),
#             dict(
#                 meta_keys=(
#                     'img_id',
#                     'img_path',
#                     'ori_shape',
#                     'img_shape',
#                     'scale_factor',
#                 ),
#                 type='PackDetInputs'),
#         ],
#         ann_file='train/_annotations.coco.json',
#         data_prefix=dict(img='train/')))

# val_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         pipeline=[
#             dict(type='LoadImageFromFile', imdecode_backend='tifffile', to_float32=True),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(type='Resize',scale_factor=1.0, keep_ratio=True),
#             dict(
#                 meta_keys=(
#                     'img_id',
#                     'img_path',
#                     'ori_shape',
#                     'img_shape',
#                     'scale_factor',
#                 ),
#                 type='PackDetInputs'),
#         ],
#         ann_file='valid/_annotations.coco.json',
#         data_prefix=dict(img='valid/')))

# # optim_wrapper = dict(
# #     optimizer=dict(
# #         capturable = True
# #     )
# # )

# # test_dataloader = val_dataloader
# test_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         pipeline=[
#             dict(type='LoadImageFromFile', imdecode_backend='tifffile', to_float32=True),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(type='Resize',scale_factor=1.0, keep_ratio=True),
#             dict(
#                 meta_keys=(
#                     'img_id',
#                     'img_path',
#                     'ori_shape',
#                     'img_shape',
#                     'scale_factor',
#                 ),
#                 type='PackDetInputs'),
#         ],
#         ann_file='test/_annotations.coco.json',
#         data_prefix=dict(img='test/')))

# val_evaluator = dict(
#     classwise = True,
#     ann_file=data_root + 'valid/_annotations.coco.json',
# )

# # test_evaluator = val_evaluator
# test_evaluator = dict(
#     classwise = True,
#     ann_file=data_root + 'test/_annotations.coco.json',
# )

# model = dict(
#     data_preprocessor=dict(
#         type='DetDataPreprocessor',
#         mean=[1.1473892, 1.1473892, 1.1473892],
#         std=[3.029705, 3.029705, 3.029705],
#         bgr_to_rgb=False,
#         pad_size_divisor=1)
# )



# default_hooks = dict(
#     checkpoint=dict(
#          interval=5,
#          max_keep_ckpts=2,  # only keep latest 2 checkpoints
#          save_best='auto'
#      ),
#      logger=dict(type='LoggerHook', interval=5))