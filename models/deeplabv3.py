from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def deeplabv3_resnet101_(device="cpu"):

    model = deeplabv3_resnet101(weights=None, progress=True, num_classes=2)
    model.classifier = DeepLabHead(2048, 1)
    model = model.to(device)
    source = "torch"
    return (model, source)

def deeplabv3_resnet101_p(device="cpu"):

    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT, progress=True)
    model.classifier = DeepLabHead(2048, 1)
    model = model.to(device)
    source = "torch"
    return (model, source)

def deeplabv3_resnet50_(device="cpu"):

    model = deeplabv3_resnet50(weights=None, progress=True, num_classes=2)
    model.classifier = DeepLabHead(2048, 1)
    model = model.to(device)
    source = "torch"
    return (model, source)

