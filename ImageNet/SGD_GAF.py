import torch
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import required
import torch.nn.functional as F


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
                 weight_decay=0, nesterov=False, alpha=1.0, beta=1.5):
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
                if alpha >=0 and beta >=0:
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
                if alpha >= 0 and beta >=0:
                    d_p = alpha * torch.atan(beta*d_p)               
                p.data.add_(-group['lr'], d_p)

        return loss
        
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