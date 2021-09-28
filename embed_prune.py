import torch
import torch.nn.utils.prune as prune
import numpy as np
from pruning_utils import prune_model_custom, pruning_model
# layer1.0.conv2.weight_mask
# layer3.1.conv1.weight_mask
torch.manual_seed(1)
from models.resnet import resnet50, resnet18
a = torch.load("resnet18_cifar10_lt_extreme_qrcode/model_SA_best.pth.tar", map_location="cpu")
model = resnet18(num_classes=10)

def extract_mask(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]
    return new_dict

mask = extract_mask(a['state_dict'])
prune_model_custom(model, mask)
model.load_state_dict(a['state_dict'])
from pruning_utils import check_sparsity
print(check_sparsity(model))
p=0.5
pruning_model(model, p)
print(check_sparsity(model))
mask_new = extract_mask(model.state_dict())
qrmask = mask_new['layer1.0.conv2.weight_mask']
qrmask = (qrmask.sum((2,3)) > 0).float().numpy()
#qrmask = qrmask[237 - 14 : 237 - 14 + 29, 128 - 14:128 - 14 + 29]
qrmask = qrmask[33:33 + 29, 12:12 + 29]
qrmask = qrmask

qrmask[1, 1:5] = 1
qrmask[5, 1:5] = 1
qrmask[1:5, 1] = 1
qrmask[1:5, 5] = 1
qrmask[7, 0:8] = 1
qrmask[0:7, 7] = 1
qrmask[-2, 1:5] = 1
qrmask[-6, 1:5] = 1
qrmask[-6:-2, 1] = 1
qrmask[-6:-2, 5] = 1
qrmask[-8, 0:7] = 1
qrmask[-8:-1,7] = 1

qrmask[1, -6:-2] = 1
qrmask[5, -6:-2] = 1
qrmask[1:5, -2] = 1
qrmask[1:5, -6] = 1

qrmask[0:7, -8] = 1
qrmask[7,-8:-1] = 1

import matplotlib.pyplot as plt
plt.imshow(qrmask)
plt.savefig(f"prune_2_{p}.png")