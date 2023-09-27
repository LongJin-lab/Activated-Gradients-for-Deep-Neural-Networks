"""
ImageNet training script.
Including APEX (distributed training), and DALI(data pre-processing using CPU+GPU) provided by NIVIDIA.
Thanks pytorch demo, implus (Xiang Li from NJUST), DALI.
Author: Xu Ma
Date: Aug/15/2019
Email: xuma@my.unt.edu

Useage:
python3 -m torch.distributed.launch --nproc_per_node=8 main_cosne.py -a old_resnet18 --b 64 --opt_level O0



"""


import argparse
import os
import shutil
import time
import math
import traceback
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import models as models
from utils import Logger, mkdir_p
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from SGD_GAF import SGD_atan, SGD_atanMom, Adam_atan
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
                     
global_step = 0
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', default='/media3/datasets/imagenet/', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-bs', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=250, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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
parser.add_argument("--alpha", default=1.0, type=float)
parser.add_argument("--beta", default=1.5, type=float)
parser.add_argument("--notes", default='', type=str)
parser.add_argument('--clip', default='False', type=str)
cudnn.benchmark = True

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

best_prec1 = 0
args = parser.parse_args()

if args.opt_level == 'O1':
    args.keep_batchnorm_fp32 = None

if args.opt_level == 'O0':
    mixed_type = 'FP32'
elif args.opt_level == 'O1':
    mixed_type = 'conser_mixed'
elif args.opt_level == 'O2':
    mixed_type = 'fast_mixed'
elif args.opt_level == 'O3':
    mixed_type = 'FP16'
    
args.n_gpu = torch.cuda.device_count()

args.save_path = 'checkpoints/imagenet/'+args.arch+'/_opt_level_'+mixed_type+\
                                    '_clip_'+args.clip+'_opt'+str(args.opt)+'Alpha'+str(args.alpha)+\
                                  'Beta'+str(args.beta)+'BS'+str(args.batch_size)+'LR'+\
                                   str(args.lr)+'n_gpu_'+str(args.n_gpu)+'epochs'+\
                                   str(args.epochs)+'warmup'+str(args.warmup)+'_'+\
                                   args.notes+\
                                    "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
                                        
# checkpoint
if args.checkpoint is None:
    # args.checkpoint='checkpoints/imagenet/'+args.arch
    args.checkpoint= args.save_path
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

# make apex optional
if args.distributed:
    print("Import APEX!")
    try:
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
        from apex import amp, optimizers
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def main():
    global best_prec1, args, writer
    writer = SummaryWriter(log_dir=args.save_path+'/Tensorboard')
    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size

    if not os.path.isdir(args.checkpoint) and args.local_rank == 0:
        mkdir_p(args.checkpoint)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


    
    # create model
    
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model = model.cuda()

    # args.lr = float(args.lr * float(args.batch_size * args.world_size) / 256.)  # default args.lr = 0.1 -> 256
    optimizer = set_optimizer(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale,
                                      verbosity = 0)

    model = DDP(model, delay_allreduce=True)


    # optionally resume from a checkpoint
    title = 'ImageNet-' + args.arch
    args.lastepoch =-1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.lastepoch = checkpoint['epoch']
            if args.local_rank == 0:
                logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.local_rank == 0:
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Valid Top5.'])

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    if(args.arch == "inception_v3"):
        crop_size = 299
        val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256

    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=traindir, crop=crop_size, dali_cpu=args.dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    pipe = HybridValPipe(batch_size=args.test_batch, num_threads=4, device_id=args.local_rank, data_dir=valdir, crop=crop_size, size=val_size)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    train_loader_len = int(train_loader._size / args.batch_size)
    if args.resume:
        scheduler = CosineAnnealingLR(optimizer, args.epochs, train_loader_len,
                                      eta_min=0., last_epoch=args.lastepoch, warmup=args.warmup)
    else:
        scheduler = CosineAnnealingLR(optimizer,
                                      args.epochs, train_loader_len, eta_min=0., warmup=args.warmup)
    total_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch

        if args.local_rank == 0:
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        [train_loss, train_acc, avg_train_time] = train(train_loader, model, criterion, optimizer, epoch, scheduler)
        total_time.update(avg_train_time)
        # evaluate on validation set
        [test_loss, prec1, prec5] = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            # append logger file
            logger.append([optimizer.param_groups[0]['lr'], train_loss, test_loss, train_acc, prec1, prec5])

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best,checkpoint=args.checkpoint)
            if epoch == args.epochs - 1:
                print('##Top-1 {0}\n'
                      '##Top-5 {1}\n'
                      '##Perf  {2}'.format(prec1, prec5, args.total_batch_size / total_time.avg))
                print('hyperparameters:', args.save_path)
        # reset DALI iterators
        train_loader.reset()
        val_loader.reset()
    print('settings', args.save_path)
    if args.local_rank == 0:
        logger.close()


def train(train_loader, model, criterion, optimizer, epoch, scheduler):
    global writer
    global global_step
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        global_step += 1
        lr = scheduler.update(epoch, i)
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        train_loader_len = int(train_loader._size / args.batch_size)

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as loss_item:
            loss_item.backward()
        if args.clip == 'norm':
            clip_grad_norm_(model.parameters(), max_norm=args.alpha, norm_type=2)
        elif args.clip == 'value':
            clip_grad_value_(model.parameters(), args.alpha*np.pi/2)
        optimizer.step()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
            print('[{0}/{1}]\t'
                  'Batch Time {batch_time.avg:.3f}\t'
                  'Data Time {data_time.avg:.3f}\t'
                  'Speed {2:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Top1 {top1.avg:.3f}\t'
                  'Top5 {top5.avg:.3f}'.format(
                i, train_loader_len,args.total_batch_size / batch_time.avg,
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5))
        
        
#        for name, param in net.named_parameters():
#            writer.add_histogram(name, param, global_step)
#            writer.add_histogram(name+'/grad', param.grad, global_step)
        writer.add_scalar('train_loss_step', to_python_float(reduced_loss), global_step)
        writer.add_scalar('train_acc_1_step', to_python_float(prec1), global_step)
        writer.add_scalar('train_acc_5_step', to_python_float(prec5), global_step)

#    for name, param in net.named_parameters():
#        writer.add_histogram(name, param, epoch)
#        writer.add_histogram(name+'/grad', param.grad, epoch)
    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_acc_1', top1.avg, epoch)
    writer.add_scalar('train_acc_5', top5.avg, epoch)        
    return [losses.avg, top1.avg, batch_time.avg]


def validate(val_loader, model, criterion, epoch):
    global best_prec, writer
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        val_loader_len = int(val_loader._size / args.batch_size)

        target = target.cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Speed {2:.3f} \t'
                  'Loss {loss.avg:.4f}\t'
                  'Top1 {top1.avg:.3f}\t'
                  'Top5 {top5.avg:.3f}'.format(
                   i, val_loader_len,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
    if args.local_rank == 0:
        print(' TEST Top1 {top1.avg:.4f} Top5 {top5.avg:.4f}'.format(top1=top1, top5=top5))
    writer.add_scalar('test_loss', losses.avg, epoch)
    writer.add_scalar('test_acc_1', top1.avg, epoch)
    writer.add_scalar('test_acc_5', top5.avg, epoch)
    return [losses.avg, top1.avg,top5.avg]


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


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
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)#correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

class CosineAnnealingLR(object):
    def __init__(self, optimizer, T_max, N_batch, eta_min=0, last_epoch=-1, warmup=0):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.T_max = T_max
        self.N_batch = N_batch
        self.eta_min = eta_min
        self.warmup = warmup

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.update(last_epoch+1)
        self.last_epoch = last_epoch
        self.iter = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            lrs = [base_lr * (self.last_epoch + self.iter / self.N_batch) / self.warmup for base_lr in self.base_lrs]
        else:
            lrs = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup + self.iter / self.N_batch) / (self.T_max - self.warmup))) / 2
                    for base_lr in self.base_lrs]
        return lrs

    def update(self, epoch, batch=0):
        self.last_epoch = epoch
        self.iter = batch + 1
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

        return lrs


def set_optimizer(model):
    if args.wdall:
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = SGD_atan(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta)
        print('weight decay on all parameters')
    else:
        no_decay_list = []
        decay_list = []
        no_decay_name = []
        decay_name = []
        for m in model.modules():
            if (hasattr(m, 'groups') and m.groups > 1) or isinstance(m, nn.BatchNorm2d) \
                    or m.__class__.__name__ == 'GL':
                no_decay_list += m.parameters(recurse=False)
                for name, p in m.named_parameters(recurse=False):
                    no_decay_name.append(m.__class__.__name__ + name)
            else:
                for name, p in m.named_parameters(recurse=False):
                    if 'bias' in name:
                        no_decay_list.append(p)
                        no_decay_name.append(m.__class__.__name__ + name)
                    else:
                        decay_list.append(p)
                        decay_name.append(m.__class__.__name__ + name)

        params = [{'params': no_decay_list, 'weight_decay': 0} \
            , {'params': decay_list}]
        #optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        #optimizer = SGD_atan(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta)  
        if args.opt == 'SGD_ori':
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif  args.opt == 'SGD_atan':
            optimizer = SGD_atan(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta)
        elif  args.opt == 'SGD_atanMom':
            optimizer = SGD_atanMom(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta)        
        elif args.opt == 'Adam_ori':
            optimizer = Adam_atan(params, betas=(0.9, 0.999), weight_decay=args.weight_decay, alpha=-1, beta=-1)
        elif args.opt == 'Adam_atan':
            optimizer = Adam_atan(params, betas=(0.9, 0.999), weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta)
    #    elif args.opt = 'RMSprop':
    #        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=5e-4, momentum=0.9)
    #    elif args.opt = 'RMSprop_atan':
    #        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=5e-4, momentum=0.9)      
    return optimizer


if __name__ == '__main__':
    main()
