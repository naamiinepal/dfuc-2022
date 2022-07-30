Parameter | Description
--- | ---
`base_dir` | Path to project directory
`train_img_dir` | Path to images for training relative to `base_dir`
`train_msk_dir` | Path to segmentation masks for training relative to `base_dir`
`val_img_dir` | Path to images for validation relative to `base_dir`
`val_msk_dir` | Path to segmentation masks for validation relative to `base_dir`
`train_batch_size` | Batch size during training
`val_batch_size` | Batch size during validation
`learning_rate` | Learning rate for training
`num_classes` | Number of classes to segment
`val_interval` | Number of epochs to run validation after
`eval_metric` | Metric for evaluation
`grad_accumulation_interval` | Number of batches to accumulate gradient during training(use to increase batch size in low resource settings)
`min_epoch` | Minimum number of epoch to run training
`max_epoch` | Maximum number of epoch to run training
`device_type` | Type of device to use for training(`cpu` or `gpu`)
`loss_function` | Loss function to use for training(available options -> [`dice_loss`, `focal_loss`, `dice_focal_loss`, `gen_dice_focal_loss`, `tversky_loss`])
`optim` | Optimizer to use for training(available options -> [`adam`])
`model_name` | Name of model to use for training(available options -> [`deeplabv3`, `unet`, `att_unet`, `unetr`])
`model_backbone_network` | Backbone network to use for model during training(for `deeplabv3` model, available options -> [`resnet50`, `resnet101`, `resnet101p`], for other models available options -> [])
`model_last_checkpoint_path` | Path to checkpoint if to be resumed from some checkpoint
`use_last_checkpoint` | Set to use some checkpoint(`0` or `1`)
`model_save_dir` | Path to save model checkpoints
`tb_log_dir` | Path to save tensorboard logs
`remarks` | Additional information to append to name training identifiers
---
**Additional parameters for training using k folds**
---
Parameter | Description
--- | ---
`num_fold` | Number of fold to divide the training dataset into
`k_fold_n` | takes value from 1 to `num_fold` to choose the fold to train
---
**Note: Make sure that you choose only one value for parameters for which available options are given as lists with one or more elements.**

