
from typing import Dict
from src.utils.weight_schedulers import LinearDecayScheduler, NoDecayScheduler, ToleranceScheduler, WeightScheduler


class WeightSchedulerFactory:
    
    
    def __init__(self):
        self.name = self.__class__.__name__
        
        
    def get_scheduler_by_name(self, weight_scheduler_dict: Dict[str, object]) -> WeightScheduler:
        name = weight_scheduler_dict['name']
        params = weight_scheduler_dict['parameters']
        
        if not 'init_weight' in params:
            raise ValueError('''The weight scheduler require an initial weight specified''')

        init_weight = params['init_weight']

        if name == 'linear_decay':           
            lower_bound = params.get('lower_bound', 0.)
            return self.__get_linear_decayer(init_weight=init_weight, lower_bound=lower_bound)
        
        elif name == 'tolerance_scheduler':
            increment_step = params.get('increment_step', .1)
            tolerance = params.get('tolerance', 1e-5)
            upper_bound = params.get('upper_bound', 1.)
            return self.__get_tolerance_scheduler(init_weight=init_weight, increment_step=increment_step,
                                                  tolerance=tolerance, upper_bound=upper_bound)
            
        elif name == 'no_decay':
            return self.__get_constant_scheduler(init_weight=init_weight)
            
    def __get_linear_decayer(self, init_weight: float, lower_bound: float = 0.):
        return LinearDecayScheduler(init_weight=init_weight, lower_bound=lower_bound)
    
    def __get_tolerance_scheduler(self, init_weight: float,
                                  increment_step: float = .1,
                                  tolerance: float = 1e-5,
                                  upper_bound: float = 1.):
        return ToleranceScheduler(init_weight=init_weight, increment_step=increment_step,
                                  tolerance=tolerance, upper_bound=upper_bound)
        
        
    def __get_constant_scheduler(self, init_weight: float):
        return NoDecayScheduler(init_weight=init_weight)