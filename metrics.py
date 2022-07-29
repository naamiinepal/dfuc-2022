from monai.metrics import DiceMetric

def get_metric(name = "dice_socre"):
    if name == "dice_score":
        return DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

