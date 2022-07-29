from monai.networks.nets import UNet
from monai.networks.nets import AttentionUnet
from monai.networks.nets import UNETR

def unet_(device="cpu"):

    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=3,
        ).to(device)
    source = "monai"
    return (model, source)

def attention_unet_(device="cpu"):

    model = AttentionUnet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(64, 128, 256, 512,),
        strides=(2, 2, 2),
        kernel_size= 3,
        dropout= 0.2
        ).to(device)
    source = "monai"
    return (model, source)

def unetr_(device="cpu"):

    model = UNETR(
            img_size=(640, 480),
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            hidden_size=384,
            mlp_dim=1536,
            num_heads=6
            ).to(device)
    source = "monai"
    return (model, source)

