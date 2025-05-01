import torch

from pathlib import Path
from data_loader import MyDataset
import pandas as pd

from mymodel import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(data_dir, out_dir=''):

    cfig = {
        'train_bs': 4,
        'val_bs': 8,
        'csv_root': data_dir / 'meta_data_test.csv',
        'scale_dose_dict': data_dir / 'PTV_DICT.json',
        'pat_obj_dict': data_dir / 'Pat_Obj_DICT.json',
        'num_workers': 4,
        'down_HU': -1000,
        'up_HU': 1000,
        'denom_norm_HU': 500,
        'in_size': [96, 128, 144],
        'out_size': [96, 128, 144],
        'norm_oar': True,
        'CatStructures': False,
        'dose_div_factor': 10
    }
    
    ds = MyDataset(cfig, 'test')
    d = ds[0]
    print(d.keys())

    # # Inference: iterate through the dataset since aug is within
    # for data_dict in ds:

    #     x = torch.tensor(data_dict['data'].numpy()).to(device)
    #     x = x.unsqueeze(0)

    #     body_mask = x[3:4,...].bool()
    #     outbody_mask = ~body_mask
    #     ptv_mask = x[1:2, ...].bool()
    #     near_mask = torch.logical_and(~ptv_mask, body_mask)

    #     info = x[-1].mean((1, 2))[:4].cpu().tolist()
    #     info = torch.tensor(info).to(device)

    #     x = x[:-1]

    #     pred = model(x)

    #     pred_ori = ds.aug.inverse(dict(img = pred))
    #     pred_ori = pred_ori.squeeze().numpy()

if __name__ == "__main__":
    data_dir = Path('data')
    inference(data_dir)
