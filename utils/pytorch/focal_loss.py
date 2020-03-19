import torch
import torch.nn as nn
import torch.nn.functional as func


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input_, target):
        if input_.dim() > 2:
            input_ = input_.view(input_.size(0), input_.size(1), -1)
            input_ = input_.transpose(1, 2)
            input_ = input_.contiguous().view(-1, input_.size(2))
        target = target.view(-1, 1)
        log_pt = func.log_softmax(input_)
        log_pt = log_pt.gather(1, target)
        log_pt = log_pt.view(-1)
        pt = log_pt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input_.data.type():
                self.alpha = self.alpha.type_as(input_.data)
            at = self.alpha.gather(0, target.data.view(-1))
            log_pt = log_pt * at

        loss = -1 * (1 - pt) ** self.gamma * log_pt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
