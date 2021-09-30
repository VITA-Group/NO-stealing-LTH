



# Environment
Our experiments are conducted with PyTorch==1.6.0

# Checkpoints for reproduce: 

Coming Soom. 

# Experiments

## ResNet-20s
### IMP
python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_0.2 --init pretrained_model/res20s_cifar100_lt.pth.tar --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 --prune_type lt --rewind_epoch 3 

### Scheme 1
python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_extreme.pth.tar --fc --num-paths 1000 --type ewp --prune-type lt

### Scheme 2 
python embed_res20s_cifar10.py

### Scheme 3 
python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger0 --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_extreme.pth.tar --fc --save_model --lr 0.1 

## ResNet-18s

# Acknowledgement

# Citation