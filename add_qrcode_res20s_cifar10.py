'''
load lottery tickets and evaluation 
support datasets: cifar10, Fashionmnist, cifar100
'''

import os
import time 
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from advertorch.utils import NormalizeByChannelMeanStd

from utils import *
from pruning_utils_2 import *
from pruning_utils_unprune import *

parser = argparse.ArgumentParser(description='PyTorch Evaluation Tickets')

##################################### general setting #################################################
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--arch', type=str, default='res18', help='model architecture')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save_model', action="store_true", help="whether saving model")

##################################### training setting #################################################
parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

##################################### Pruning setting #################################################
parser.add_argument('--pretrained', default=None, type=str, help='pretrained weight for pt')
parser.add_argument('--mask_dir', default=None, type=str, help='mask direction for ticket')
parser.add_argument('--conv1', action="store_true", help="whether pruning&rewind conv1")
parser.add_argument('--fc', action="store_true", help="whether rewind fc")

parser.add_argument('--type', type=str, default=None, choices=['ewp', 'random_path', 'betweenness', 'hessian_abs', 'taylor1_abs','intgrads','identity'])
parser.add_argument('--add-back', action="store_true", help="add back weights")
parser.add_argument('--prune-type', type=str, choices=["lt", 'pt', 'st', 'mt', 'trained', 'transfer'])

parser.add_argument('--evaluate-p', type=float, default=0.00)
parser.add_argument('--evaluate-random', action='store_true')

parser.add_argument('--max-name', type=str)
parser.add_argument('--checkpoint', type=str)



best_sa = 0

def main():
    global args, best_sa
    args = parser.parse_args()
    print(args)

    print('*'*50)
    print('conv1 included for prune and rewind: {}'.format(args.conv1))
    print('fc included for rewind: {}'.format(args.fc))
    print('*'*50)
    torch.manual_seed(2)
    np.random.seed(2)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset 
    model, train_loader, val_loader, test_loader = setup_model_dataset(args)
    
    criterion = nn.CrossEntropyLoss()
    

    state_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
    current_mask = extract_mask(state_dict)
    masks = []
    for name in current_mask:
        masks.append(current_mask[name].view(-1))
    
    masks = torch.cat(masks, 0)
    masks[masks == 0] = torch.rand((masks == 0).sum())
    masks[masks < 1 - args.evaluate_p] = 0
    masks[masks > 1 - args.evaluate_p] = 1
    count = 0
    for name in current_mask:
        current_mask[name] = masks[count:count+current_mask[name].nelement()].view(current_mask[name].shape)
        count += current_mask[name].nelement()
        
        state_dict[name[:-12] + '.weight_orig'] = torch.where(state_dict[name] == current_mask[name], state_dict[name[:-12] + '.weight_orig'], torch.randn(state_dict[name[:-12] + '.weight_orig'].shape) * 0.1)
        state_dict[name] = current_mask[name]
    prune_model_custom(model, current_mask, conv1=False)
    check_sparsity(model, conv1=False)
    try:
        model.load_state_dict(state_dict)
    except:
        state_dict['normalize.mean'] = model.state_dict()['normalize.mean']
        state_dict['normalize.std'] = model.state_dict()['normalize.std']
        model.load_state_dict(state_dict)

    #validate(val_loader, model, criterion)
    check_sparsity(model, conv1=False)
    #if args.evaluate_p > 0:
    #    pruning_model(model, args.evaluate_p, random=args.evaluate_random)

    validate(test_loader, model, criterion)
    check_sparsity(model, conv1=False)
    state = model.state_dict()

    mask = state[args.max_name + ".weight_mask"].clone()
    mask = mask.sum((2,3)) > 0
    mask = mask.int().numpy()
    plt.imshow(mask)
    plt.savefig(f'ownership/{args.arch}_{args.dataset}_qrcode_add_{args.evaluate_p}.png')
    #assert False
    mask = mask[3:3+29, 1:1+29]
    #assert False
    import qrcode
    qr = qrcode.QRCode(
        version=3,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=1,
        border=0,
    )
    qr.add_data('signature')
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    code = np.array(img)

    mask[0:9, 0:9] = code[0:9, 0:9]
    mask[-9:,:9] = code[-9:,:9]
    mask[:9,-9:] = code[:9,-9:]
    mask[20:25, 20:25] = code[20:25, 20:25]
    mask[-8, 4 * 3 + 9] = 1
    mask[6] = code[6]
    mask[:, 6] = code[:, 6]
    print((mask != code).mean())

    plt.imshow(mask)
    plt.savefig(f'ownership/{args.arch}_{args.dataset}_qrcode_add_{args.evaluate_p}.png')
    assert False
    remove_prune(model, False)
    
    validate(test_loader, model, criterion)
    prune_model_custom(model, current_mask)
    model.load_state_dict(state_dict)
    check_sparsity(model, False)
    validate(test_loader, model, criterion)
    


def save_checkpoint(state, is_SA_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SA_best.pth.tar'))
def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()
        

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def setup_seed(seed): 
    torch.manual_seed(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

if __name__ == '__main__':
    main()


