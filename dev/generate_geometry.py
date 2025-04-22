import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage.io
from easydict import EasyDict
from utils import *

from pathlib import Path

data_dir = Path('/workspaces/GDP-HMM_AAPMChallenge')
data_path = data_dir / 'data/0617-259694+imrt+MOS_33896.npz'
data_npz = np.load(data_path, allow_pickle=True)
d = EasyDict(data_npz['arr_0'].item())
print(d.keys())

def save_fig(arr, filename, cmap='gray'):
    plt.tight_layout(pad=0)
    plt.imshow(arr, cmap)
    plt.axis('off')
    plt.savefig(filename)

save_fig(d.img[48], 'ct-48.png', 'gray')
save_fig(d.dose[48], 'dose-48.png', 'hot')

beam_plate1 = get_per_beamplate(
    PTV_mask = d['PTV'].copy(),
    isocenter = d['isocenter'],
    space = [2, 2.5, 2.5],
    gantry_angle = d['angle_list'][0],
    with_distance = True
  ).astype('float')

save_fig(beam_plate1[48]+d.PTV[48], 'plate1.png', 'hot')
