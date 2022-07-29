from models.deeplabv3 import deeplabv3_resnet101_, deeplabv3_resnet50_, deeplabv3_resnet101_p
from models.unet import unet_, attention_unet_, unetr_

def load_model(model_name, model_backbone_network, device):
    if model_name == "deeplabv3" and model_backbone_network == "resnet101":
        model, source = deeplabv3_resnet101_(device)
    elif model_name == "deeplabv3" and model_backbone_network == "resnet50":
        model, source = deeplabv3_resnet50_(device)
    elif model_name == "deeplabv3" and model_backbone_network == "resnet101p":
        model, source = deeplabv3_resnet101_p(device)
    elif model_name == "unet":
        model, source = unet_(device)
    elif model_name == "att_unet":
        model, source = attention_unet_(device)
    elif model_name == "unetr":
        model, source = unetr_(device)
    else:
        print("Please provide valid model name and model backbone network")
        return -1
    return model, source

