'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler
import time

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tensorboardX import SummaryWriter
from GAF import SGD_atan, SGD_atanMom, Adam_atan, SGD_atanMom_Ada
from Adam import Adam_atan
from RMSprop import RMSprop

from models_cifar100 import *
from datetime import datetime
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
# Training

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
parser.add_argument('--epoch', type=int, default=200, help='training epoch')
parser.add_argument('--warm', type=int, default=1, help='warm up training phase')

                                    
parser.add_argument('--data', default='./data', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', '-a', default='ResNet18', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-bs', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize (default: 200)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=250, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# Optimization options
parser.add_argument('--opt_level', default='O2', type=str,
                    help='O2 is fast mixed FP16/32 training, O0 (FP32 training) and O3 (FP16 training), O1 ("conservative mixed precision"), O2 ("fast mixed precision").--opt_level O1 and O2 both use dynamic loss scaling by default unless manually overridden. --opt-level O0 and O3 (the "pure" training modes) do not use loss scaling by default. See more in https://github.com/NVIDIA/apex/tree/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet')
parser.add_argument('--keep-batchnorm-fp32', default=True, action='store_true',
                    help='keeping cudnn bn leads to fast training')
parser.add_argument('--loss-scale', type=float, default=None)
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('-t', '--test', action='store_true',
                    help='Launch test mode with preset arguments')
parser.add_argument('--warmup', '--wp', default=5, type=int,
                    help='number of epochs to warmup')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 4e-5 for mobile models)')
parser.add_argument('--wd-all', dest = 'wdall', action='store_true',
                    help='weight decay on all parameters')
parser.add_argument('--opt', default='SGD_atan', type=str)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--ex", default=0, type=int)
parser.add_argument("--alpha", default=0., type=float)
parser.add_argument("--beta", default=0., type=float)
parser.add_argument("--shrink", default=0., type=float)
parser.add_argument("--gamma", default=0., type=float)
parser.add_argument("--notes", default='', type=str)
parser.add_argument('--clip', default='False', type=str)
args = parser.parse_args()

args.save_path = 'runs/'+args.dataset+'/'+args.arch+'/Alpha'+str(args.alpha)+\
                                  'Beta'+str(args.beta)+'Shrink'+str(args.shrink)+\
                                  'Gamma'+str(args.gamma)+'_clip_'+args.clip+str(args.opt)+\
                                    '_BS'+str(args.batch_size)+'LR'+\
                                   str(args.lr)+'epoch'+\
                                   str(args.epoch)+'warmup'+str(args.warm)+\
                                   args.notes+\
                                    "{0:%Y-%m-%dT%H-%M/}".format(datetime.now())
def train(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()
    net.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        loss.backward()
        if args.clip == 'norm':
            clip_grad_norm_(net.parameters(), max_norm=args.alpha, norm_type=2)
        elif args.clip == 'value':
            clip_grad_value_(net.parameters(), args.alpha)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: {:.1f}, Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, losses.avg, top1.avg))
    writer.add_scalar('Train/Average loss', losses.avg, epoch)
    writer.add_scalar('Train/Accuracy-top1', top1.avg, epoch)
    writer.add_scalar('Train/Accuracy-top5', top5.avg, epoch)
    writer.add_scalar('Train/Time', batch_time.sum, epoch)
    #for name, param in net.named_parameters():
        #layer, attr = os.path.splitext(name)
        #attr = attr[1:]
        #writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    return top1.avg, losses.avg, batch_time.sum


def test(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()
    net.eval()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            # test_loss += loss.item()
            # _, preds = outputs.max(1)
            # correct += preds.eq(old_labels).sum()
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', losses.avg, epoch)
    writer.add_scalar('Test/Accuracy-top1', top1.avg, epoch)
    writer.add_scalar('Test/Accuracy-top5', top5.avg, epoch)
    writer.add_scalar('Test/Time', batch_time.sum, epoch)
    
    return top1.avg, losses.avg, batch_time.sum
            


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
        
        
if __name__ == '__main__':
  
#if 1:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    # Data
    if args.dataset == 'cifar10':
        #from models_cifar10 import *
        print('==> Preparing cifar10 data..')
      
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
        trainset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, args.batch_size, shuffle=True, num_workers=args.workers)
    
        testset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, args.batch_size, shuffle=False, num_workers=args.workers)
    
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
                   
    elif args.dataset == 'cifar100':
        #from models_cifar100 import *
        print('==> Preparing cifar100 data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    
        trainset = torchvision.datasets.CIFAR100(
            root='./data/cifar100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, args.batch_size, shuffle=True, num_workers=args.workers)
    
        testset = torchvision.datasets.CIFAR100(
            root='./data/cifar100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, args.batch_size, shuffle=False, num_workers=args.workers)
    
#        classes = ('plane', 'car', 'bird', 'cat', 'deer',
#                   'dog', 'frog', 'horse', 'ship', 'truck')
    
    
    # Model
    print('==> Building model..')
    #net = VGG('VGG16')
    #net_name = "ResNet50"
    #model_name = "ResNet50"
    net_name = args.arch
    model_name = args.arch

    if args.arch == "ResNet18" :
        net = ResNet18()
    elif args.arch=="VGG16" :
        net = VGG('VGG16')
    elif args.arch=="LeNet" :
        net = LeNet()        
    elif args.arch=="SigmoidLeNet" :
        net = SigmoidLeNet()  
    elif args.arch=="ResNet152" :
        net = ResNet152()   
    elif args.arch=="ResNet50" :
        net = ResNet50()   
    elif args.arch=="MobileV1" :
        net = MobileNet()         
    elif args.arch=="MobileV2" :
        net = MobileNetV2() 
    elif args.arch=="RegNetX_200MF" :
        net = RegNetX_200MF() 
    elif args.arch=="EfficientNetB0" :
        net = EfficientNetB0()
    elif args.arch=="densenet_cifar" :
        net = densenet_cifar()
        
    elif args.arch=="ResNet34" :
        net = ResNet34() 
    elif args.arch=="ResNet101" :
        net = ResNet101()   
    elif args.arch=="ShuffleNetV2" :
        net = ShuffleNetV2(net_size=0.5)
    elif args.arch=="GoogLeNet" :
        net = GoogLeNet()
    elif args.arch=="ResNeXt29_2x64d" :
        net = ResNeXt29_2x64d()        
    elif args.arch=="SENet18" :
        net = SENet18()
    elif args.arch=="PreActResNet18" :
        net = PreActResNet18()        
    elif args.arch=="PreActResNet34" :
        net = PreActResNet34()        
    elif args.arch=="PreActResNet50" :
        net = PreActResNet50()        
    elif args.arch=="DPN26" :
        net = DPN26()
    elif args.arch=="ShuffleNetG2" :
        net = ShuffleNetG2()        
    elif args.arch=="DenseNet121" :
        net = DenseNet121()          
    elif args.arch=="DenseNet169" :
        net = DenseNet169()         
        
                                   
    #net = ResNet50()
    #net = PreActResNet18()
    # net = GoogLeNet()
    #net = DenseNet121()
    #net = ResNeXt29_2x64d()
    #net = MobileNet()
    #net = MobileNetV2()
    #net = DPN92()
    #net = ShuffleNetG2()
    #net = SENet18()
    #net = ShuffleNetV2(1)
    #net = EfficientNetB0()
    #net = RegNetX_200MF()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    if args.opt == 'SGD_ori':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif  args.opt == 'SGD_atan':
        optimizer = SGD_atan(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, alpha=args.alpha, beta=args.beta)
    elif  args.opt == 'SGD_atanMom':
        optimizer = SGD_atanMom(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, alpha=args.alpha, beta=args.beta)      
    elif  args.opt == 'SGD_atanMom_Ada':
        optimizer = SGD_atanMom_Ada(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, shrink=args.shrink, gamma=args.gamma)           
    elif args.opt == 'Adam_ori':
        optimizer = Adam_atan(net.parameters(), betas=(0.9, 0.999), weight_decay=5e-4, alpha=-1, beta=-1)
    elif args.opt == 'Adam_atan':
        optimizer = Adam_atan(net.parameters(), betas=(0.9, 0.999), weight_decay=5e-4, alpha=args.alpha, beta=args.beta)
#    elif args.opt = 'RMSprop':
#        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=5e-4, momentum=0.9)
#    elif args.opt = 'RMSprop_atan':
#        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=5e-4, momentum=0.9)    
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    iter_per_epoch = len(trainloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    train_time  = 0.0
    test_time = 0.0
    train_top1_acc = 0.0
    train_min_loss = 100
    test_top1_acc = 0.0
    test_min_loss = 100
    #lr_list = []
    

    writer = SummaryWriter(log_dir=args.save_path)


    for epoch in range(1, args.epoch):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        #lr_list.append(optimizer.param_groups[0]['lr'])
        train_acc_epoch, train_loss_epoch, train_epoch_time = train(epoch)
#        if epoch > args.warm:
#            train_scheduler.step(epoch)
        train_top1_acc = max(train_top1_acc, train_acc_epoch)
        train_min_loss = min(train_min_loss, train_loss_epoch)
        train_time += train_epoch_time
        acc, test_loss_epoch, test_epoch_time = test(epoch)
        test_top1_acc = max(test_top1_acc, acc)
        test_min_loss = min(test_min_loss, test_loss_epoch)
        test_time += test_epoch_time


    writer.close()
    end_train = train_time//60
    end_test = test_time//60
    print(model_name)
    print("train time: {}D {}H {}M".format(end_train//1440, (end_train%1440)//60, end_train%60))
    print("tset time: {}D {}H {}M".format(end_test//1440, (end_test%1440)//60, end_test%60))
    print("train_acc_top1:{}, train_min_loss:{}, train_time:{}, test_top1_acc:{}, test_min_loss:{}, test_time:{}".format(train_top1_acc, train_min_loss, train_time, test_top1_acc, test_min_loss, test_time))
    print("0.3-4.5")
    print("args.save_path:", args.save_path)
