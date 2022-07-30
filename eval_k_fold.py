import os
from glob import glob
import csv
import torch
import yaml
import numpy as np
from PIL import Image
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.metrics import get_confusion_matrix, compute_confusion_matrix_metric
from transforms import val_imtrans_d, post_trans, val_trans_d_no_gt, val_trans_d_aug
from load_models import load_model
from data_loader import data_loader_d, data_loader_d_k_fold

with open('./config/eval_config_k_fold.yaml', 'r') as config_file:
    eval_config_params = yaml.safe_load(config_file)

def main():
    eval_name = eval_config_params["eval_name"]
    model_name = eval_config_params["model_name"]
    model_backbone_network = eval_config_params["model_backbone_network"]
    model_checkpoint_path = eval_config_params["model_checkpoint_path"]
    output_base_dir = eval_config_params["output_base_dir"]
    eval_type = eval_config_params["eval_type"]
    eval_data_dir_0 = eval_config_params["eval_data_dir_0"]
    eval_data_base_dir_1 = eval_config_params["eval_data_base_dir_1"]
    images_path = eval_config_params["images_path"]
    masks_path = eval_config_params["masks_path"]
    eval_batch_size = eval_config_params["eval_batch_size"]
    k_fold_n = eval_config_params["k_fold_n"]
    num_fold = eval_config_params["num_fold"]

    op_dir = os.path.join(output_base_dir, eval_name)

    if eval_type == 0:
        op_prefix = "external_valid"
    elif eval_type == 1:
        op_prefix = "internal_valid"

    op_dir = op_dir + "_" + op_prefix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(op_dir):
        os.mkdir(op_dir)

    eval_images = []
    false_positive_error = 0
    false_negative_error = 0
    precision_metric = 0
    recall_metric = 0

    if eval_type == 0:
        eval_images = sorted(glob(os.path.join(eval_data_dir_0, "*.jpg")))
        eval_loader = data_loader_d(val_trans_d_no_gt(), eval_batch_size, eval_data_dir_0)
   
    elif eval_type == 1:        
        image_fname = []
        mask_fname = []

        with open ("k_fold-splits.csv", "r") as infile:
            for image in infile:
                image_fname = image.split(",")[:-1]
        for image in image_fname:
            mask_fname.append((image.replace("image", "mask")).replace("jpg", "png"))
        
        eval_images = image_fname
        eval_loader= data_loader_d_k_fold(val_trans_d_aug(), eval_batch_size, image_fname[(k_fold_n-1)*len(image_fname)//num_fold:(k_fold_n*len(image_fname)//num_fold)], mask_fname[(k_fold_n-1)*len(image_fname)//num_fold:(k_fold_n*len(image_fname)//num_fold)])
    
    fname_mapping_list = []
    for img in eval_images:
        fname_mapping_list.append(img.split('/')[-1])

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    post_infer_trans = post_trans()

    model, source = load_model(model_name, model_backbone_network, device)
    model.load_state_dict(torch.load(model_checkpoint_path)["model_state"])
  
    model.eval()

    with torch.no_grad():

        idx = 0
        list_for_csv = []

        for eval_data in eval_loader:
    
            eval_images = eval_data["img"].to(device)
            if eval_type == 1:
                eval_masks = eval_data["msk"].to(device)
            
            if source == "monai":
                roi_size = (640, 480)
                sw_batch_size = 2
                eval_outputs = sliding_window_inference(eval_images, roi_size, sw_batch_size, model)
                eval_outputs = [post_infer_trans(i) for i in decollate_batch(eval_outputs)]

            elif source == "torch":
                eval_outputs = model(eval_images)
                eval_outputs = post_trans()(list(eval_outputs.items())[0][1])          

            if eval_type == 1:
                dice_score = dice_metric(y_pred=eval_outputs, y=eval_masks)      
                false_positive_error = compute_confusion_matrix_metric("fpr", get_confusion_matrix(torch.Tensor(eval_outputs[0]).reshape(1, 1, 640, 480), torch.Tensor(eval_masks)))
                false_negative_error = compute_confusion_matrix_metric("fnr", get_confusion_matrix(torch.Tensor(eval_outputs[0]).reshape(1, 1, 640, 480), torch.Tensor(eval_masks)))
                precision_metric = compute_confusion_matrix_metric("precision", get_confusion_matrix(torch.Tensor(eval_outputs[0]).reshape(1, 1, 640, 480), torch.Tensor(eval_masks)))
                recall_metric = compute_confusion_matrix_metric("recall", get_confusion_matrix(torch.Tensor(eval_outputs[0]).reshape(1, 1, 640, 480), torch.Tensor(eval_masks)))
                      
            for eval_output in eval_outputs:
                Image.fromarray(eval_output[0].cpu().detach().numpy().astype("uint8") * 255).transpose(Image.Transpose.TRANSPOSE).save(os.path.join(op_dir,fname_mapping_list[idx].split(".")[0]+".png"))
                if eval_type == 1:
                    list_for_csv.append([fname_mapping_list[idx].split(".")[0] + ".png", dice_score, false_positive_error, false_negative_error, precision_metric, recall_metric])
                idx = idx + 1

        # aggregate the final mean dice result
        if eval_type == 1:
            
            print("Final evaluation metric:", dice_metric.aggregate().item())

            with open('outputs/csv_files/dice_scores_all' + eval_name  + '.csv', 'w') as csv_file:
                writer_csv = csv.writer(csv_file)
                for items in list_for_csv:
                    writer_csv.writerow([items[0], items[1].item(), items[2].item(), items[3].item(), items[4].item(), items[5].item()])

if __name__ == "__main__":
    main()
        