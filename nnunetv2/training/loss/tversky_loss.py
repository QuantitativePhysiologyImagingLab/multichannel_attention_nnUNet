import torch
from torch import nn

# ---- Add Tversky loss class ----
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, target):
        # logits: (B, C, X, Y, Z), target: (B, 1, X, Y, Z)
        p = torch.softmax(logits, dim=1)[:, 1:]  # foreground probs
        y = (target == 1).float()
        TP = (p * y).sum(dim=list(range(2, p.ndim)))
        FP = (p * (1 - y)).sum(dim=list(range(2, p.ndim)))
        FN = ((1 - p) * y).sum(dim=list(range(2, p.ndim)))
        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        return (1. - tversky).pow(self.gamma).mean()
