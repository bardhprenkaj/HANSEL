from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Dict

import numpy as np


class WeightScheduler(ABC):
    
    def __init__(self, init_weight):
        self.name = self.__class__.__name__
        self.init_weight = init_weight
        self.first_weight = init_weight
        
    @abstractmethod
    def update(self, args: Dict[str, object], **kwargs) -> float:
        pass
    
    
    def reset(self):
        self.init_weight = self.first_weight
    
    
class NoDecayScheduler(WeightScheduler):
    
    def __init__(self, init_weight):
        super().__init__(init_weight)
    
    def update(self, args: Dict[str, object], **kwargs):
        return self.init_weight
    
class LinearDecayScheduler(WeightScheduler):
    
    def __init__(self, init_weight, lower_bound=0):
        super().__init__(init_weight)
        
        self.lower_bound = lower_bound
        
    def update(self, args, **kwargs):
        args = SimpleNamespace(**args)
        curr_epoch = args.curr_epoch
        overall_epochs = args.overall_epochs
        # Define a schedule for alpha
        alpha_schedule = np.linspace(self.init_weight, self.lower_bound, num=overall_epochs)
        alpha = alpha_schedule[curr_epoch]
        return alpha
    
class ToleranceScheduler(WeightScheduler):
    
    def __init__(self,
                 init_weight,
                 increment_step=.1,
                 tolerance=1e-5,
                 upper_bound=1.):
        super().__init__(init_weight)
        
        self.increment_step = increment_step
        self.tolerance = tolerance
        self.upper_bound = upper_bound
        
        self.EPS = 1e-3
        
    def update(self, args, **kwargs):
        args = SimpleNamespace(**args)
        prev_loss = args.prev_loss
        curr_loss = args.curr_loss
        
        if prev_loss is None:
            return self.init_weight

        # calculate the change in loss
        delta = np.abs(curr_loss - prev_loss)
        # return the new weight
        if delta <= self.tolerance:
            self.init_weight = min(self.init_weight + self.increment_step, self.upper_bound)
        else:
            self.init_weight += self.EPS
        return self.init_weight
