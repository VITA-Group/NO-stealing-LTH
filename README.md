# You are caught stealing my winning lottery ticket! Making a lottery ticket claim its ownership

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Codes for [NeurIPS'21] [You are caught stealing my winning lottery ticket! Making a lottery ticket claim its ownership]().

Xuxi Chen*, Tianlong Chen*, Zhenyu Zhang, Zhangyang Wang

## Overall Story

The lottery ticket hypothesis emerges as a promising framework to leverage a special sparse subnetwork (i.e., *winning ticket*) instead of a full model for both training and inference, that can lower both costs without scarifying the performance. The main resource bottleneck of LTH is however the extraordinary cost to find the sparse mask of the winning ticket. That makes the found winning ticket become a valuable asset to the owners, highlighting the necessity of protecting its copyright. 

Our setting adds a new dimension to the recently soaring interest in protecting against the intellectual property (IP) infringement of deep models and verifying their ownerships, since they take owners' resources to develop or train. While existing methods explored encrypted weights or predictions, we investigate a unique way to leverage sparse topological information to perform *lottery verification*, by developing several graph-based signatures that can be embedded as credentials. By further combining trigger set-based methods, our proposal can work in both white-box and black-box verification scenarios. Specifically, our verification is shown to be robust to removal attacks such as model fine-tuning and pruning, as well as several ambiguity attacks.

## Environment
PyTorch 1.6.0

## Checkpoints for reproduce: 


## Experiments
[Extreme ticket of res20](https://drive.google.com/file/d/1-0zIyGeRvCwYU0HQtwssilSbjI01ANCi/view?usp=sharing).
[epoch_3.pth.tar](https://drive.google.com/file/d/1yVsB5bxL6WVSE-H1v4QN60vgaaM98uF9/view?usp=sharing)
Put it under `ownership/`
### ResNet-20s
#### IMP
python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_0.2 --init pretrained_model/res20s_cifar100_lt.pth.tar --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 --prune_type lt --rewind_epoch 3 

#### Scheme 1
python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_extreme.pth.tar --fc --num-paths 1000 --type ewp --prune-type lt

#### Scheme 2 
```bash
python embed_res20s_cifar10.py # embed
python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_qr --pretrained ownership/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --prune-type lt --type identity --save_model # 
python extract_qrcode_res20s_cifar10.py --save_dir _ --data datasets/cifar10 --checkpoint res20s_cifar10_lt_qr/checkpoint.pth.tar --arch res20s --max-name layer2.1.conv1 --start-x 2 --start-y 3
```

#### Scheme 3 
python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger0 --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_extreme.pth.tar --fc --save_model --lr 0.1 


## Citation
```
```
