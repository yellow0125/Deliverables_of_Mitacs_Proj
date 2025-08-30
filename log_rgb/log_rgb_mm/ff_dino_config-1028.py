# The new config inherits a base config to highlight the necessary modification
_base_ = './mmdetection/configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(num_classes=4)
)

resume = True

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        capturable = True
    )
)
optimizer=dict(capturable = True)

# Modify dataset related settings
data_root = '../LogRGB-Forest-Data-1/'
metainfo = {
    'classes': ("Alive Tree", "Debris", "Dead Tree", "Beetle Fire Tree"),
    'palette': [
        (70, 230, 70),
        (255, 250, 0),
        (255, 69, 0),
        (255, 5, 5)
    ]
}
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/')))

# Modify metric related settings
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox',
    classwise=True,  # Enable per-class evaluation
    format_only=False)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric='bbox',
    classwise=True,  # Enable per-class evaluation
    format_only=False)

# Modify evaluation settings
evaluation = dict(
    metric=['bbox'],
    classwise=True,  # Enable per-class evaluation
    interval=1)  # Evaluate every epoch

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=2,  # only keep latest 2 checkpoints
        save_best='auto'
    ),
    early_stopping = dict(
        type="EarlyStoppingHook",
        patience=10,
        monitor="coco/bbox_mAP",
        min_delta=0.005))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,  # Override the base config's epochs
    val_interval=1
)
runner = dict(type='EpochBasedRunner', max_epochs=300)

work_dir="/scratch/yang.zhiyu/tif_test/ff_dino_1028"



# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'