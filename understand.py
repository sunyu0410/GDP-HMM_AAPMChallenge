import yaml
import json
import torch
import pandas as pd

from pathlib import Path
from data_loader import MyDataset, InputDataset

prj_dir = Path('/workspaces/GDP-HMM_AAPMChallenge')

cfig = yaml.load(open(prj_dir / 'data/config.yaml'), Loader=yaml.FullLoader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = MyDataset(cfig['loader_params'], 'train')
ds2 = InputDataset(cfig['loader_params'], 'train')

x = ds[0]
x2 = ds2[0]
