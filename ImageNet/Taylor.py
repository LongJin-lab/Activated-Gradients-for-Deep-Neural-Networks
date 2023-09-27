from torch.optim.optimizer import Optimizer, required
import torch as t

class Taylor(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Taylor, self).__init__(params, defaults)
        self.history_params = []

    def __setstate__(self, state):
        super(Taylor, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            
    def step(self, epoch, sub_epoch, step, closure=None):
        loss = None
        if closure is not None:
            with t.enable_grad():
                loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            if epoch < sub_epoch:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0: # 进行正则化
                        # add_表示原处改变，d_p = d_p + weight_decay*p.data
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p] # 之前的累计的数据，v(t-1)
                        # 进行动量累计计算
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = t.clone(d_p).detach()
                        else:
                            # 之前的动量
                            buf = param_state['momentum_buffer']
                            # buf= buf*momentum + （1-dampening）*d_p
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov: # 使用neterov动量
                            # d_p= d_p + momentum*buf
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf
                    # p = p - lr*d_p
                    p.data.add_(-group['lr'], d_p)
            else:
                if step < 3:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        d_p = p.grad.data
                        if weight_decay != 0: # 进行正则化
                            # add_表示原处改变，d_p = d_p + weight_decay*p.data
                            d_p.add_(weight_decay, p.data)
                        if momentum != 0:
                            param_state = self.state[p] # 之前的累计的数据，v(t-1)
                            # 进行动量累计计算
                            if 'momentum_buffer' not in param_state:
                                buf = param_state['momentum_buffer'] = t.clone(d_p).detach()
                            else:
                                # 之前的动量
                                buf = param_state['momentum_buffer']
                                # buf= buf*momentum + （1-dampening）*d_p
                                buf.mul_(momentum).add_(1 - dampening, d_p)
                            if nesterov: # 使用neterov动量
                                # d_p= d_p + momentum*buf
                                d_p = d_p.add(momentum, buf)
                            else:
                                d_p = buf
                        param_state = self.state[p]
                        if step == 0:
                            param_state['w_2'] = t.clone(p).detach()
                        if step == 1:
                            param_state['w_1'] = t.clone(p).detach()
                        if step == 2:
                            param_state['w'] = t.clone(p).detach()
                        p.data.add_(d_p, alpha=-group['lr'])
                else:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        d_p = p.grad.data
                        if weight_decay != 0: # 进行正则化
                            # add_表示原处改变，d_p = d_p + weight_decay*p.data
                            d_p.add_(weight_decay, p.data)
                        if momentum != 0:
                            param_state = self.state[p] # 之前的累计的数据，v(t-1)
                            # 进行动量累计计算
                            if 'momentum_buffer' not in param_state:
                                buf = param_state['momentum_buffer'] = t.clone(d_p).detach()
                            else:
                                # 之前的动量
                                buf = param_state['momentum_buffer']
                                # buf= buf*momentum + （1-dampening）*d_p
                                buf.mul_(momentum).add_(1 - dampening, d_p)
                            if nesterov: # 使用neterov动量
                                # d_p= d_p + momentum*buf
                                d_p = d_p.add(momentum, buf)
                            else:
                                d_p = buf
                        param_state = self.state[p]
                        w = param_state['w']
                        w_1 = param_state['w_1']
                        w_2 = param_state['w_2']
                        
                        theta_1 = 1.5 * w
                        theta_2 = -w_1
                        theta_3 = 0.5 * w_2 
                        theta_4 = group['lr'] * d_p
                        # theta_4 = group['lr'] * p.grad
                        p.data = theta_1 + theta_2 + theta_3 - theta_4

                        param_state['w_2'] = t.clone( w_1 ).detach()
                        param_state['w_1'] = t.clone( w ).detach()
                        param_state['w'] = t.clone( p ).detach()
        return loss