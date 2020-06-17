import torch


class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                print(len(state))
                
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

                state['step'] = 0
                state['square_avg'] = torch.zeros_like(p.data)
                state['square_avg'].share_memory_()
                if group['momentum'] > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['momentum_buffer'].share_memory_()
                if group['centered']:
                    state['grad_avg'] = torch.zeros_like(p.data)
                    state['grad_avg'].share_memory_()