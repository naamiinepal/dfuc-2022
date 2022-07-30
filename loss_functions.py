from monai.losses import DiceLoss, DiceFocalLoss, GeneralizedDiceFocalLoss, FocalLoss, TverskyLoss
import random 

def get_loss_function(name="dice_loss"):
    if name == "dice_loss":
        return DiceLoss(sigmoid=True)
    elif name == "focal_loss":
        return FocalLoss(sigmoid=True)
    elif name == "dice_focal_loss":
        return DiceFocalLoss(sigmoid=True)
    elif name == "gen_dice_focal_loss":
        return GeneralizedDiceFocalLoss(sigmoid=True)
    elif name == "tversky_loss":
        b = random.uniform(0.9, 1)
        a = 1 - b
        return TverskyLoss(sigmoid=True, alpha=a, beta=b)
        