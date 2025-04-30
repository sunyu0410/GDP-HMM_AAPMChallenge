import yaml
import json
import torch
import pandas as pd

from pathlib import Path
from data_loader import MyDataset

prj_dir = Path('/workspaces/GDP-HMM_AAPMChallenge')

cfig = yaml.load(open(prj_dir / 'data/config.yaml'), Loader=yaml.FullLoader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = MyDataset(cfig['loader_params'], 'train')
ds2 = MyDataset(cfig['loader_params'], 'train')

x = ds[0]
x2 = ds2[0]

for key in x:
    print(key, torch.all(torch.tensor(x[key]==x2[key])))