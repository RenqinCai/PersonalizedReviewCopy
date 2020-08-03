import torch.optim as optim

class _OPTIM:
    def __init__(self, params, args):
        super().__init__()
        optimizer_type = args.optimizer
        lr = args.learning_rate
        momentum = args.momentum
        # weight_decay = args.weight_decay
        # eps = args.eps

        if optimizer_type == "RMSProp":
            self.m_optimizer = optim.RMSProp(params, lr=lr,  momentum=momentum)

        elif optimizer_type == "Adam":
            self.m_optimizer = optim.Adam(params, lr=lr)

        else:
            raise NotImplementedError

    def zero_grad(self):
        self.m_optimizer.zero_grad()

    def step(self):
        self.m_optimizer.step()