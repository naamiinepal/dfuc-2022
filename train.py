import logging
import os
import sys
import math
import yaml
import json
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import monai
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.visualize import plot_2d_or_3d_image
from transforms import post_trans

from transforms import (
    train_trans_d_aug,
    val_trans_d_aug
    )

from data_loader import (
    data_loader_d
    )

from load_models import load_model
from loss_functions import get_loss_function
from metrics import get_metric
from optimizers import get_optimizer

#load configs
with open('./config/train_config.yaml', 'r') as config_file:
    config_params = yaml.safe_load(config_file)
    model_config = json.dumps(config_params)

def main():

    #init configs
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    base_dir = config_params["base_dir_path"]["base_dir"]
    train_img_dir = config_params["data_dir_path"]["train_img_dir"]
    train_msk_dir = config_params["data_dir_path"]["train_msk_dir"]
    val_img_dir = config_params["data_dir_path"]["val_img_dir"]
    val_msk_dir = config_params["data_dir_path"]["val_msk_dir"]
    train_batch_size = config_params["training_params"]["train_batch_size"]
    val_batch_size = config_params["training_params"]["val_batch_size"]
    learning_rate = config_params["training_params"]["learning_rate"]
    loss_function_name = config_params["training_params"]["loss_function"]
    eval_metric_name = config_params["training_params"]["eval_metric"]
    optim_name = config_params["training_params"]["optim"]
    num_classes = config_params["training_params"]["num_classes"]
    val_interval = config_params["training_params"]["val_interval"]
    grad_accumulation_interval = config_params["training_params"]["grad_accumulation_interval"]
    min_epoch = config_params["training_params"]["min_epoch"]
    max_epoch = config_params["training_params"]["max_epoch"]
    device_type = config_params["training_params"]["device_type"]
    tensorboard_log_dir = config_params["logs_params"]["tensorboard_logs_params"]["tb_log_dir"]
    model_save_dir = config_params["model_params"]["model_save_dir"]
    model_name = config_params["model_params"]["model_name"]
    model_backbone_network = config_params["model_params"]["model_backbone_network"]
    last_checkpoint = config_params["model_params"]["model_last_checkpoint_path"]
    use_last_checkpoint = config_params["model_params"]["use_last_checkpoint"]

    checkds = monai.utils.misc.first(data_loader_d(train_trans_d_aug(), train_batch_size, os.path.join(base_dir, train_img_dir), os.path.join(base_dir, train_msk_dir)))

    training_name = str(model_name) + "_" + str(model_backbone_network) + "_" + str(loss_function_name) + "_" + str(optim_name) + "_" + str(learning_rate) + "_" + str(train_batch_size * grad_accumulation_interval) + "_" + str(datetime.now()).replace(' ', '-')

    if not os.path.exists(base_dir + model_save_dir + training_name):
            os.mkdir(base_dir + model_save_dir + training_name )

    with open(base_dir + model_save_dir + training_name + "/" + training_name + ".json", "w") as outfile:
        outfile.write(model_config)

    #set device
    if device_type == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if device == torch.device("cuda"):
        print("Training running in device: " + torch.cuda.get_device_name())
    elif device == torch.device("cpu"):
        print("Training running in device: cpu")
    else:
        print("Device is neither cpu nor gpu")

    #load model 
    model, source = load_model(model_name, model_backbone_network, device)

    #training parameters

    loss_function = get_loss_function(loss_function_name)
    optimizer = get_optimizer(optim_name, model, float(learning_rate))
    eval_metric = get_metric(eval_metric_name)

    best_metric= -1
    best_metric_epoch = -1
    best_n_metric = [-1, -1, -1, -1, -1]
    best_n_metric_epoch = [-1, -1, -1, -1, -1]
    epoch_loss_values = list()
    metric_values = list()
    training_metric = 0
    flag_train = 1
    epoch = 0
    count = 0
    previous_metric = -math.inf

    writer = SummaryWriter(base_dir + tensorboard_log_dir + training_name + '/')

    if use_last_checkpoint:
        checkpoint= torch.load(os.path.join(base_dir, "checkpoints/" + last_checkpoint))
        epoch= checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])

    train_loader= data_loader_d(train_trans_d_aug(), train_batch_size, os.path.join(base_dir, train_img_dir), os.path.join(base_dir, train_msk_dir))
    val_loader= data_loader_d(val_trans_d_aug(), val_batch_size, os.path.join(base_dir, val_img_dir), os.path.join(base_dir, val_msk_dir))

    y_threshold = Variable(torch.Tensor([0.5]).to(device))

    #training loop
    while (epoch < max_epoch and flag_train):
        print("-" * 25)
        print(f"epoch {epoch + 1}/{max_epoch}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, masks = batch_data["img"].to(device), batch_data["msk"].to(device)

            if (step - 1) % grad_accumulation_interval == 0:
                optimizer.zero_grad()

            outputs = model(inputs)

            if source == "torch":
                outputs = list(outputs.items())[0][1]

            loss = loss_function(outputs, masks)
            loss.backward()

            if (step - 1) % grad_accumulation_interval == 0:
                optimizer.step()

            epoch_loss += loss.item()

            eval_metric((torch.sigmoid(outputs) >= y_threshold).float() * 1, masks)

            epoch_len = 1800 // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        writer.add_scalar("train_loss_per_epoch", epoch_loss, epoch + 1)

        training_metric = eval_metric.aggregate().item()
        eval_metric.reset()

        writer.add_scalar("train mean dice", training_metric, epoch + 1)

        print(f"epoch {epoch + 1} average loss: {epoch_loss: .4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                k = 0
                for val_data in val_loader:
                    val_images, val_masks = val_data["img"].to(device), val_data["msk"].to(device)
                    if source == "monai":
                        roi_size=(480, 480)
                        sw_batch_size=2
                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                        val_outputs = [post_trans
                        ()(i) for i in decollate_batch(val_outputs)]
                    elif source == "torch":
                        val_outputs = model(val_images)
                        val_outputs = post_trans()(list(val_outputs.items())[0][1])

                    # compute metric for current iteration
                    eval_metric(y_pred=val_outputs, y=val_masks)
                    k = k + 1

                    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                    plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                    plot_2d_or_3d_image(val_masks, epoch + 1, writer, index=0, tag="label")
                    plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")
                # aggregate the final mean dice result
                metric = eval_metric.aggregate().item()
                # reset the status for next validation round
                eval_metric.reset()
                metric_values.append(metric)

                # check_metric sees if the new validation score is greater than any of the best n scores
                check_metric = np.multiply(metric > np.array(best_n_metric), 1)

                if np.sum(check_metric) >= 1:

                    break_flag = 0

                    for best_val_idx in range(0, len(best_n_metric)):

                        if metric >= best_n_metric[best_val_idx] and break_flag == 0:

                            for i_bv in range(best_val_idx, len(best_n_metric) - 1):

                                curr_idx = len(best_n_metric) + best_val_idx - i_bv

                                best_n_metric[curr_idx - 1] = best_n_metric[curr_idx - 2]
                                best_n_metric_epoch[curr_idx - 1] = best_n_metric_epoch[curr_idx - 2]

                                if os.path.exists(base_dir + model_save_dir + training_name + "/" + training_name + "_best_" +str(curr_idx - 2) + ".pth"):
                                    os.rename(base_dir + model_save_dir + training_name + "/" + training_name + "_best_" + str(curr_idx - 2) + ".pth", base_dir + model_save_dir + training_name + "/" + training_name + "best" + str(curr_idx - 1) + ".pth")

                            best_n_metric[best_val_idx] = metric
                            best_n_metric_epoch[best_val_idx] = epoch + 1
                            best_metric = best_n_metric[0]
                            best_metric_epoch = best_n_metric_epoch[0]

                            torch.save({"model_state": model.state_dict(), "model_config":model_config,"epoch": epoch+1, "optim_state": optimizer.state_dict()}, base_dir + model_save_dir + training_name + "/" + training_name + "best" + str(best_val_idx) + ".pth")
                            print("saved new best metric model")
                            break_flag = 1

                    count = 0

                if metric <= previous_metric:
                    count += 1

                previous_metric = metric

                torch.save({"model_state": model.state_dict(), "model_config":model_config,"epoch": epoch + 1, "optim_state": optimizer.state_dict()}, base_dir + model_save_dir + training_name + "/" + training_name + "_checkpoint.pth")
                print("saved last checkpoint")

                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                writer.add_scalar("best_val_mean_dice_epoch", best_metric, epoch + 1)
                writer.add_scalar("best_val_mean_dice_epoch", best_metric_epoch, epoch + 1)

            epoch += 1

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

if __name__ == "__main__":
    main()

