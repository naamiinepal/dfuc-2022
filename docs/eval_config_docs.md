Parameter | Description
--- | ---
`eval_name` | Name of the evaluation
`model_name` | Name of model to use for evaluation(available options -> [`deeplabv3`, `unet`, `att_unet`, `unetr`])
`model_backbone_network` | Backbone network to use for model during training(for `deeplabv3` model, available options -> [`resnet50`, `resnet101`, `resnet101p`], for other models available options -> [])
`model_checkpoint_path` | Path of checkpoint to run evaluation
`output_base_dir` | Path to base directory to store outputs
`eval_type` | Evaluation type based on whether ground truth of evaluation data is present or not(`0` or `1`)
`eval_data_dir_0` | If `eval_type=0`, directory of data to use for evaluation
`eval_data_base_dir_1` | If `eval_type=1`, base directory of data to use for evaluation
`images_path` | If `eval_type=1`, path of image data to use for evaluation relative to `eval_data_base_dir`
`masks_path` | If `eval_type=1`, path of segmentations data to use for evaluation relative to `eval_data_base_dir`
`eval_batch_size` | Batch size during evaluation
---
**Note: Make sure that you choose only one value for parameters for which available options are given as lists with one or more elements.**

