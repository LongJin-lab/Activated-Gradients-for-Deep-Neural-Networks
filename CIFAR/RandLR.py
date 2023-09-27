#import types
import math
#from torch._six import inf
#from functools import wraps
import warnings
#import weakref
#from collections import Counter
#from bisect import bisect_right
import random
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class RandLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, RandType, last_epoch=-1, verbose=False):
        #self.step_size = step_size
        #self.gamma = gamma
        
        #optimizer, RandType, last_epoch, verbose)
        super(RandLR, self).__init__( optimizer , last_epoch, verbose)
        #self.optimizer = optimizer
        self.RandType = RandType
        #self.last_epoch = last_epoch
        #self.verbose = verbose
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0) :
            return [group['lr'] for group in self.optimizer.param_groups]
        if self.RandType == 'uniform':
            RandFac = random.uniform(0, 1)
            for group in self.optimizer.param_groups:
                print('LR:', group['initial_lr'] * RandFac)
        elif self.RandType == 'Beta':
            RandFac = random.betavariate(2, 2)
        return [group['initial_lr'] * RandFac
                for group in self.optimizer.param_groups]

#    def _get_closed_form_lr(self):
#        return [group['lr'] * RandFac
#                for base_lr in self.base_lrs]    
#        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
#                for base_lr in self.base_lrs]