import torch
from math import pi, cos

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0002, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + cos(pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs] 