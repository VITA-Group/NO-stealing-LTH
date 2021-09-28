import torch
import numpy as np
from models.resnets import resnet20
a = torch.load("ownership/res18_cifar10_extreme.pth.tar", map_location="cpu")
a.keys()
import sys
def check_sparsity(mask, conv1=True):
    
    sum_list = 0
    zero_sum = 0
    for name in mask:
        if 'mask' in name:
            mask_ = mask[name]
            sum_list = sum_list+float(mask_.nelement())
            zero_sum = zero_sum+float(torch.sum(mask_ == 0))    

    print(1 - zero_sum / sum_list)

np.random.seed(2)
def extract_mask(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]
    return new_dict

mask = extract_mask(a['state_dict'])
check_sparsity(mask)
import qrcode
qr = qrcode.QRCode(
    version=3,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=1,
    border=0,
)
qr.add_data('signature')
qr.make()

img = qr.make_image(fill_color="black", back_color="white")
code = np.array(img)
from scipy.signal import correlate2d
h,w = code.shape[0],code.shape[1]
max_sim = 0
for name in mask:
    if not 'layer1' in name:
        continue
    mask_ = mask[name].sum((2,3)).numpy() > 0
    mask_ = mask_.astype(float)
    if (mask_.shape[0] - code.shape[0] < 0) or (mask_.shape[1] - code.shape[1] < 0):
        continue
    sim = np.zeros((mask_.shape[0] - code.shape[0] + 1, mask_.shape[1] - code.shape[1] + 1))
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            sim[i,j] = (mask_[i:i+h,j:j+w] == code).mean()

    if np.max(sim) > max_sim:
        max_name = name
        max_sim = np.max(sim)
print(max_name)
print(max_sim)
#max_name = 'layer2.0.conv2.weight_mask' # override
import sys
if len(sys.argv) > 1:
    max_name = sys.argv[1]
print(mask.keys())
print(max_name)
mask_ = mask[max_name].sum((2,3)).numpy() > 0
mask_ = mask_.astype(float)
sim = np.zeros((mask_.shape[0] - code.shape[0] + 1, mask_.shape[1] - code.shape[1] + 1))
for i in range(sim.shape[0]):
    for j in range(sim.shape[1]):
        sim[i,j] = (mask_[i:i+h,j:j+w] == code).mean()
r, c = np.where(sim == np.max(sim))

r = r[0]
c = c[0]
print(r,c)
real_mask = mask[max_name].numpy()[r:r+h, c:c+w].copy()
real_mask_one = (real_mask == 1).sum()
real_mask_flat = ((real_mask).sum((2,3)) > 0).astype(float)
print(real_mask_flat.shape)



for i in range(code.shape[0]):
    for j in range(code.shape[1]):
        if code[i,j] == 1 and real_mask_flat[i,j] == 0:
            _ = np.array([0] * 9)
            _[0] = 1
            new_mask = np.random.permutation(_)
            real_mask[i,j] = new_mask.reshape((3, 3))
            real_mask_flat[i,j] == 1
        elif code[i,j] == 0 and real_mask_flat[i,j] == 1:
            real_mask[i,j] = 0
            real_mask_flat[i,j] == 0

real_mask_one_new = (real_mask == 1).sum()
real_mask_flat_new = (real_mask).sum((2,3))
diff = real_mask_one_new - real_mask_one
print(diff)
if (diff > 0):
    # remove some connections
    real_mask_flat_greater_0 = np.where(real_mask_flat_new > 1)
else:
    # recover some connections
    pos = np.expand_dims((code == 1), (2, 3)) * np.expand_dims(real_mask_flat == 1, (2,3)) * (real_mask == 0)
    pos = np.where(pos)
    pos = np.stack(pos)
    print(pos.shape)
    pos = pos[:, np.random.permutation(pos.shape[1])[:(-diff)]]
    print(pos.shape)
    for i in range(pos.shape[1]):
        p = pos[:, i]
        real_mask[p[0], p[1], p[2], p[3]] = 1


import matplotlib.pyplot as plt


mask[max_name][r:r+h, c:c+w] = torch.from_numpy(real_mask)


vis = mask[max_name].sum((2,3)).numpy() > 0
plt.imshow(vis)
plt.savefig(f"ownership/res18_cifar10_vis_{max_name}.pdf")
plt.close()
torch.save(mask, f'ownership/res18_cifar10_qrcode_{max_name}.pth.tar')


check_sparsity(mask)
'''
vis = mask[max_name].sum((2,3)).numpy() > 0
plt.imshow(vis)
plt.savefig("vis2.png")
plt.close()
torch.save(mask, 'ownership/res18_cifar10_extreme.pth.tar')
'''