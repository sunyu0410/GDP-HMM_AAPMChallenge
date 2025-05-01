import sys
sys.path.append('mednext')
from nnunet_mednext import create_mednext_v1
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_mednext_v1(
  num_input_channels = 7,
  num_classes = 1,
  model_id = 'B',
  kernel_size = 3,
  deep_supervision = False
).to(device)

# model.load_state_dict(torch.load(''))