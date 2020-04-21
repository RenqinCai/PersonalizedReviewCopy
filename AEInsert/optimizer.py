import torch.optim as optim

class Optimizer:
    def __init__(self, params, args):
        optimizer_type = args.optimizer_type
        lr = args.learning_rate
        momentum = args.momentum
        weight_decay = args.weight_decay
        eps = args.eps

        if optimizer_type == "RMSProp":
            self.m_optimizer = optim.RMSProp(params, lr=lr, eps=eps, weight_decay=weight_decay, momentum=momentum)

        elif optimizer_type == "Adam":
            self.m_optimizer = optim.Adam(params, lr=lr)

        else:
            raise NotImplementedError

    def zero_grad(self):
        self.m_optimizer.zero_grad()

    def step(self):
        self.m_optimizer.step()