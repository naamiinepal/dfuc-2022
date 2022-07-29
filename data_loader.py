import os
from glob import glob
import torch
from monai.data import Dataset

from monai.data.utils import pad_list_data_collate
from torch.utils.data import DataLoader

def get_files_d(img_data_dir, msk_data_dir):

    images = sorted(glob(os.path.join(img_data_dir, "*.jpg")))
    masks = sorted(glob(os.path.join(msk_data_dir, "*.png")))

    return [{"img": img, "msk": msk} for img, msk in zip(images, masks)]

def get_files_d_no_gt(img_data_dir):

    images = sorted(glob(os.path.join(img_data_dir, "*.jpg")))

    return [{"img": img} for img in zip(images)]

# define array dataset, data loader
def data_loader_d(trans, bs, img_data_dir, msk_data_dir=None):

    if msk_data_dir:
        data_file = get_files_d(img_data_dir, msk_data_dir)
    else:
        data_file = get_files_d_no_gt(img_data_dir)

    ds = Dataset(data_file, transform=trans)

    return DataLoader(ds, batch_size=bs, num_workers=1, pin_memory=torch.cuda.is_available(), collate_fn=pad_list_data_collate)

