import math
import torch
import torch.nn as nn
import torch.optim as optim


class Yogi(optim.Optimizer):
    r"""Implements Yogi Optimizer Algorithm.

    It has been proposed in `Adaptive methods for Nonconvex Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        initial_accumulator (float, optional): initial values for first and
            second moments (default: 1e-6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. Yogi\: Adaptive methods for Nonconvex Optimization:
        https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization
    """

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-3,
                 initial_accumulator=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= initial_accumulator:
            raise ValueError("Invalid initial accumulator value: {}".format(initial_accumulator))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, initial_accumulator=initial_accumulator,
                        weight_decay=weight_decay)
        super(Yogi, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Yogi, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if(closure is not None):
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if(p.grad is None):
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if(len(state) == 0):
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = nn.init.constant_(torch.empty_like(p.data,
                                                          memory_format=torch.preserve_format),
                                                          group["initial_accumulator"])
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = nn.init.constant_(torch.empty_like(p.data,
                                                             memory_format=torch.preserve_format),
                                                             group["initial_accumulator"])

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                beta1_correction = 1 - beta1 ** state['step']
                beta2_correction = 1 - beta2 ** state['step']
                stepsize  = group['lr'] * math.sqrt(beta2_correction) / beta1_correction

                if(group['weight_decay'] != 0):
                    grad = grad.add(group['weight_decay'], p.data)

                grad2 = grad * grad

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.addcmul_((beta2 - 1), torch.sign(exp_avg_sq - grad2), grad2)

                # Update parameters for next step
                p.data.addcdiv_(-stepsize, exp_avg, torch.sqrt(exp_avg_sq).add_(group['eps']))

        return loss
