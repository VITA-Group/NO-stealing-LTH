# You are caught stealing my winning lottery ticket! Making a lottery ticket claim its ownership

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Codes for [Preprint] [You are caught stealing my winning lottery ticket! Making a lottery ticket claim its ownership]().

Xuxi Chen*, Tianlong Chen*, Zhenyu Zhang, Zhangyang Wang

## Overall Results



## Environment
PyTorch 1.6.0

## Checkpoints for reproduce: 

Coming Soom. 

## Experiments
### ResNet-20s
#### IMP
python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_0.2 --init pretrained_model/res20s_cifar10_lt.pth.tar --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 --prune_type lt --rewind_epoch 3 

python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_0.1 --init res20s_cifar10_lt_0.2/epoch_3.pth.tar --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --prune_type lt --rewind_epoch 3 --resume `last_checkpoint`

python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_0.05 --init res20s_cifar10_lt_0.2/epoch_3.pth.tar --seed 1 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --prune_type lt --rewind_epoch 3 --resume `last_checkpoint`

python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_0.01 --init res20s_cifar10_lt_0.2/epoch_3.pth.tar --seed 1 --lr 0.1 --fc --rate 0.01 --pruning_times 15 --prune_type lt --rewind_epoch 3 --resume `last_checkpoint`


#### Scheme 1
python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_extreme.pth.tar --fc --num-paths 1000 --type ewp --prune-type lt

#### Scheme 2 
python embed_res20s_cifar10.py

#### Scheme 3 
python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger0 --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_extreme.pth.tar --fc --save_model --lr 0.1 

### ResNet-18
python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_0.2 --init pretrained_model/res18_cifar10_lt.pth.tar --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 --prune_type lt --rewind_epoch 3 


## Citation
```
```

## Acknowledgement