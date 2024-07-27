import torch
import math
from copy import deepcopy

# Stochastic Weight Averaging

class SimpleSWA:
    def __init__(self, model, swa_start, swa_lr, cycle_length):
        self.model = model
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.cycle_length = cycle_length
        self.swa_model = None
        self.n_averaged = 0

    def update(self, epoch):
        if epoch >= self.swa_start:
            if self.swa_model is None:
                self.swa_model = deepcopy(self.model)
            elif ((epoch - self.swa_start) % self.cycle_length) == (self.cycle_length - 1):
                self._update_avg_model()
                self.n_averaged += 1
                print("SWA model updated")

    def _update_avg_model(self):
        for p_swa, p_model in zip(self.swa_model.parameters(), self.model.parameters()):
            device = p_swa.device
            p_swa.data.mul_(self.n_averaged / (self.n_averaged + 1.0))
            p_swa.data.add_(p_model.data.to(device) / (self.n_averaged + 1.0))

    # sawtooth learning rate schedule
    # At the begining of each cycle it increases to the swa_lr, then decreases linearly to the base_lr
    def get_lr(self, epoch, base_lr):
        if epoch < self.swa_start:
            return base_lr
        else:
            t = ((epoch - self.swa_start) % self.cycle_length) / ( self.cycle_length - 1)
            return base_lr + (self.swa_lr - base_lr)  * max(0, (1 - t))

    def get_final_model(self):
        return self.swa_model if self.swa_model is not None else self.model