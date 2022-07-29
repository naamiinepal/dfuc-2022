from monai.transforms import (
    Activations,
    AsDiscrete,
    AsDiscreted,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    EnsureType,
    EnsureTyped,
    RandSpatialCropd,
    RandRotate90d,
    RandZoomd,
    RandRotated,
    RandAxisFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    EnsureChannelFirstd,
)

# define transforms for image and segmentation
def train_imtrans_d():
    return Compose(
        [
        LoadImaged(keys=["img", "msk"]),
        EnsureChannelFirstd(keys=["img", "msk"]),
        ScaleIntensityd(keys=["img", "msk"], minv=0, maxv=1),
        AsDiscreted(keys=["msk"], threshold=0.5),
        EnsureTyped(keys=["img", "msk"])
        ]
    )

def val_imtrans_d():
    return Compose(
        [
        LoadImaged(keys=["img", "msk"]),
        EnsureChannelFirstd(keys=["img", "msk"]),
        ScaleIntensityd(keys=["img", "msk"], minv=0, maxv=1),
        AsDiscreted(keys=["msk"], threshold=0.5),
        EnsureTyped(keys=["img", "msk"])
        ]
    )

def train_trans_d_aug():
    return Compose(
        [
        LoadImaged(keys=["img", "msk"]),
        EnsureChannelFirstd(keys=["img", "msk"]),
        RandGaussianNoised(keys=["img"], prob=0.2, mean=0, std=5),
        RandGaussianSmoothd(keys=["img"], prob=0.2,sigma_x=(2, 4)),
        RandZoomd(keys=["img", "msk"] ,prob=0.2, min_zoom=0.6, max_zoom=1),
        RandRotate90d(keys=["img", "msk"] ,prob=0.3,spatial_axes=(0, 1)),
        RandAxisFlipd(keys=["img", "msk"] ,prob=0.3),
        RandRotated(keys=["img", "msk"], range_x=(0.174), prob=0.2, padding_mode="zeros"),
        RandAdjustContrastd(keys=["img"], prob=0.2, gamma=(0.5, 4)),
        ScaleIntensityd(keys=["img", "msk"], minv=0, maxv=1),
        AsDiscreted(keys=["msk"], threshold=0.5),
        EnsureTyped(keys=["img", "msk"])
        ]
    )

def val_trans_d_aug():
    return Compose(
        [
        LoadImaged(keys=["img", "msk"]),
        EnsureChannelFirstd(keys=["img", "msk"]),
        ScaleIntensityd(keys=["img", "msk"], minv=0, maxv=1),
        AsDiscreted(keys=["msk"], threshold=0.5),
        EnsureTyped(keys=["img", "msk"]),
        ]
    )

def val_trans_d_no_gt():
    return Compose(
        [
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        ScaleIntensityd(keys=["img"], minv=0, maxv=1),
        EnsureTyped(keys=["img"]),
        ]
    )

def post_trans():
    return Compose(
        [
        EnsureType(),
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5)
        ]
    )

