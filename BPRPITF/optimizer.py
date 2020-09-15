import torch.optim as optim

class _OPTIM:
    def __init__(self, params, args):
        super().__init__()
        optimizer_type = args.optimizer
        lr = args.learning_rate
        momentum = args.momentum
        weight_decay = args.weight_decay
        # weight_decay = 0.0
        # eps = args.eps

        if optimizer_type == "RMSProp":
            self.m_optimizer = optim.RMSProp(params, lr=lr,  momentum=momentum)

        elif optimizer_type == "Adam":
            self.m_optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "SGD":
            self.m_optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.m_optimizer.zero_grad()

    def step(self):
        self.m_optimizer.step()