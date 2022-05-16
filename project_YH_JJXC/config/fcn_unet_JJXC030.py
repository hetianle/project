_base_ = [
    '../../configs/_base_/models/fcn_unet_s5-d16.py', './datasets/JJCH-030.py',
    '../../configs/_base_/default_runtime.py', '../../configs/_base_/schedules/schedule_20k.py'
]
model = dict(test_cfg=dict(mode='whole'))
load_from = '/mnt/lustre/hetianle/.cache/torch/checkpoints/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth'
# load_from = '/mnt/lustrenew2/hetianle/IGSNRR/project/work_dirs/fcn_unet_building150/latest.pth'
# checkpoint_config = dict(by_epoch=False, interval=100)
# evaluation = dict(interval=100, metric='mIoU', pre_eval=True)
