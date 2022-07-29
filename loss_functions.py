from monai.losses import DiceLoss, DiceFocalLoss, GeneralizedDiceFocalLoss, FocalLoss, TverskyLoss

def get_loss_function(name="dice_loss"):
    if name == "dice_loss":
        return DiceLoss(sigmoid=True)
    elif name == "focal_loss":
        return FocalLoss(sigmoid=True)
    elif name == "gen_dice_focal_loss":
        return GeneralizedDiceFocalLoss(sigmoid=True)

