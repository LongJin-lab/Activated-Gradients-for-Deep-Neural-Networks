import torch
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import required
import torch.nn.functional as F
import math

class SGD_ori(Optimizer):


    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_ori, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_ori, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:

                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)

                    else:
                        d_p = buf
                print ('d_p_max: ', d_p.max().max().max())
                p.data.add_(-group['lr'], d_p)

        return loss

class SGD_atan(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}

        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, alpha=0.3, beta=4.5, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, alpha=alpha, beta=beta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_atan, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_atan, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            alpha = group['alpha']
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print(d_p.max().max())
                #d_p = 0.05 * torch.atan(d_p*1.5)
                #d_p = 0.3 * torch.atan(d_p*4.5)
                #d_p = 0.05 * torch.atan(d_p*1.5)
                #d_p = 0.7 * torch.atan(d_p*0.7)
                #d_p = 1.5 * (1 / (1 + torch.exp(-1 * d_p)) - 0.5)  + 0.1 * torch.sign(d_p)
                #1.5 * (1 / (1 + np.exp(-1 * x)) - 0.5) + 0.1 * np.sign(x)
                d_p = alpha * torch.atan(beta*d_p)
                #d_p = d_p.max().max()/2 * torch.atan(d_p*(1.5*2/d_p.max().max()))
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:

                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)

                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss


class SGD_atanMom(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, alpha=0.3, beta=4.5, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, alpha=alpha, beta=beta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_atanMom, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_atanMom, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            alpha = group['alpha']
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print(d_p.max().max())
                #d_p = 0.05 * torch.atan(d_p*1.5)
                #d_p = 0.3 * torch.atan(d_p*4.5)
                #d_p = 0.05 * torch.atan(d_p*1.5)
                #d_p = 0.7 * torch.atan(d_p*0.7)
                #d_p = 1.5 * (1 / (1 + torch.exp(-1 * d_p)) - 0.5)  + 0.1 * torch.sign(d_p)
                #1.5 * (1 / (1 + np.exp(-1 * x)) - 0.5) + 0.1 * np.sign(x)
                
                #d_p = d_p.max().max()/2 * torch.atan(d_p*(1.5*2/d_p.max().max()))
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:

                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)

                    else:
                        d_p = buf
                d_p = alpha * torch.atan(beta*d_p)
                p.data.add_(-group['lr'], d_p)

        return loss
        
class SGD_tanh_Mom(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, alpha=0.3, beta=4.5, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, alpha=alpha, beta=beta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_tanh_Mom, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_tanh_Mom, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            alpha = group['alpha']
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print(d_p.max().max())
                #d_p = 0.05 * torch.atan(d_p*1.5)
                #d_p = 0.3 * torch.atan(d_p*4.5)
                #d_p = 0.05 * torch.atan(d_p*1.5)
                #d_p = 0.7 * torch.atan(d_p*0.7)
                #d_p = 1.5 * (1 / (1 + torch.exp(-1 * d_p)) - 0.5)  + 0.1 * torch.sign(d_p)
                #1.5 * (1 / (1 + np.exp(-1 * x)) - 0.5) + 0.1 * np.sign(x)
                
                #d_p = d_p.max().max()/2 * torch.atan(d_p*(1.5*2/d_p.max().max()))
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:

                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)

                    else:
                        d_p = buf
                d_p = alpha * torch.tanh(beta*d_p)
                p.data.add_(-group['lr'], d_p)

        return loss        

class SGD_log_Mom(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, alpha=0.3, beta=4.5, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, alpha=alpha, beta=beta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_log_Mom, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_log_Mom, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            alpha = group['alpha']
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print(d_p.max().max())
                #d_p = 0.05 * torch.atan(d_p*1.5)
                #d_p = 0.3 * torch.atan(d_p*4.5)
                #d_p = 0.05 * torch.atan(d_p*1.5)
                #d_p = 0.7 * torch.atan(d_p*0.7)
                #d_p = 1.5 * (1 / (1 + torch.exp(-1 * d_p)) - 0.5)  + 0.1 * torch.sign(d_p)
                #1.5 * (1 / (1 + np.exp(-1 * x)) - 0.5) + 0.1 * np.sign(x)
                
                #d_p = d_p.max().max()/2 * torch.atan(d_p*(1.5*2/d_p.max().max()))
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:

                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)

                    else:
                        d_p = buf
                d_p = beta * d_p        
                d_p = alpha * torch.log( torch.relu(d_p)+1) - alpha * torch.log( torch.relu(-d_p)+1)
                p.data.add_(-group['lr'], d_p)
        return loss          
        
class SGD_atanMom_Ada(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, shrink=0, gamma=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, shrink=shrink, gamma=gamma, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_atanMom_Ada, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_atanMom_Ada, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            shrink = group['shrink']
            gamma = group['gamma']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print(d_p.max().max())
                #d_p = 0.05 * torch.atan(d_p*1.5)
                #d_p = 0.3 * torch.atan(d_p*4.5)
                #d_p = 0.05 * torch.atan(d_p*1.5)
                #d_p = 0.7 * torch.atan(d_p*0.7)
                #d_p = 1.5 * (1 / (1 + torch.exp(-1 * d_p)) - 0.5)  + 0.1 * torch.sign(d_p)
                #1.5 * (1 / (1 + np.exp(-1 * x)) - 0.5) + 0.1 * np.sign(x)
                
                #d_p = d_p.max().max()/2 * torch.atan(d_p*(1.5*2/d_p.max().max()))
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:

                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)

                    else:
                        d_p = buf
                #d_p = alpha * torch.atan(beta*d_p)
                alpha = d_p.max().max()*2/ (math.pi*shrink)
                beta = gamma/alpha
                d_p = alpha * torch.atan(beta*d_p)
                p.data.add_(-group['lr'], d_p)

        return loss        
                
class SGD_atanMom_norm(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, shrink=0, gamma=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, shrink=shrink, gamma=gamma, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_atanMom_Ada, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_atanMom_Ada, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            shrink = group['shrink']
            gamma = group['gamma']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print(d_p.max().max())
                #d_p = 0.05 * torch.atan(d_p*1.5)
                #d_p = 0.3 * torch.atan(d_p*4.5)
                #d_p = 0.05 * torch.atan(d_p*1.5)
                #d_p = 0.7 * torch.atan(d_p*0.7)
                #d_p = 1.5 * (1 / (1 + torch.exp(-1 * d_p)) - 0.5)  + 0.1 * torch.sign(d_p)
                #1.5 * (1 / (1 + np.exp(-1 * x)) - 0.5) + 0.1 * np.sign(x)
                
                #d_p = d_p.max().max()/2 * torch.atan(d_p*(1.5*2/d_p.max().max()))
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:

                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)

                    else:
                        d_p = buf
                #d_p = alpha * torch.atan(beta*d_p)
                alpha = d_p.max().max()*2/ (math.pi*shrink)
                beta = gamma/alpha
                #d_p = alpha * torch.atan(beta*d_p.*abs(p.data)/torch.norm(p.data) )
                p.data.add_(-group['lr'], d_p)

        return loss        
        
                        
#import torch
#from . import _functional as F
#from .optimizer import Optimizer
#
#
#class Adam(Optimizer):
#    r"""Implements Adam algorithm.
#
#    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
#    The implementation of the L2 penalty follows changes proposed in
#    `Decoupled Weight Decay Regularization`_.
#
#    Args:
#        params (iterable): iterable of parameters to optimize or dicts defining
#            parameter groups
#        lr (float, optional): learning rate (default: 1e-3)
#        betas (Tuple[float, float], optional): coefficients used for computing
#            running averages of gradient and its square (default: (0.9, 0.999))
#        eps (float, optional): term added to the denominator to improve
#            numerical stability (default: 1e-8)
#        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
#            algorithm from the paper `On the Convergence of Adam and Beyond`_
#            (default: False)
#
#    .. _Adam\: A Method for Stochastic Optimization:
#        https://arxiv.org/abs/1412.6980
#    .. _Decoupled Weight Decay Regularization:
#        https://arxiv.org/abs/1711.05101
#    .. _On the Convergence of Adam and Beyond:
#        https://openreview.net/forum?id=ryQu7f-RZ
#    """
#
#    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
#                 weight_decay=0, amsgrad=False):
#        if not 0.0 <= lr:
#            raise ValueError("Invalid learning rate: {}".format(lr))
#        if not 0.0 <= eps:
#            raise ValueError("Invalid epsilon value: {}".format(eps))
#        if not 0.0 <= betas[0] < 1.0:
#            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#        if not 0.0 <= betas[1] < 1.0:
#            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#        if not 0.0 <= weight_decay:
#            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#        defaults = dict(lr=lr, betas=betas, eps=eps,
#                        weight_decay=weight_decay, amsgrad=amsgrad)
#        super(Adam, self).__init__(params, defaults)
#
#    def __setstate__(self, state):
#        super(Adam, self).__setstate__(state)
#        for group in self.param_groups:
#            group.setdefault('amsgrad', False)
#
#    @torch.no_grad()
#    def step(self, closure=None):
#        """Performs a single optimization step.
#
#        Args:
#            closure (callable, optional): A closure that reevaluates the model
#                and returns the loss.
#        """
#        loss = None
#        if closure is not None:
#            with torch.enable_grad():
#                loss = closure()
#
#        for group in self.param_groups:
#            params_with_grad = []
#            grads = []
#            exp_avgs = []
#            exp_avg_sqs = []
#            state_sums = []
#            max_exp_avg_sqs = []
#            state_steps = []
#
#            for p in group['params']:
#                if p.grad is not None:
#                    params_with_grad.append(p)
#                    if p.grad.is_sparse:
#                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
#                    grads.append(p.grad)
#
#                    state = self.state[p]
#                    # Lazy state initialization
#                    if len(state) == 0:
#                        state['step'] = 0
#                        # Exponential moving average of gradient values
#                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                        # Exponential moving average of squared gradient values
#                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                        if group['amsgrad']:
#                            # Maintains max of all exp. moving avg. of sq. grad. values
#                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#
#                    exp_avgs.append(state['exp_avg'])
#                    exp_avg_sqs.append(state['exp_avg_sq'])
#
#                    if group['amsgrad']:
#                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
#
#                    # update the steps for each param group update
#                    state['step'] += 1
#                    # record the step after step update
#                    state_steps.append(state['step'])
#
#            beta1, beta2 = group['betas']
#            F.adam(params_with_grad,
#                   grads,
#                   exp_avgs,
#                   exp_avg_sqs,
#                   max_exp_avg_sqs,
#                   state_steps,
#                   group['amsgrad'],
#                   beta1,
#                   beta2,
#                   group['lr'],
#                   group['weight_decay'],
#                   group['eps'])
#        return loss

class Adam_atan(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, alpha=0.1, beta=15.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,weight_decay=weight_decay, alpha=alpha, beta=beta)
        super(Adam_atan, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if beta >= 0:
                    grad = alpha * torch.atan(grad*beta)
                    #grad = 0.05 * torch.atan(grad*1.5)
                    #grad = 0.7 * torch.atan(grad*0.7)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
