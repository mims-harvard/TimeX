import torch

def lower_bound_performance(lower_bound):
    def func(metric, sparsity):
        if metric >= lower_bound:
            return (1 - sparsity)
        return 0
    
    return func

def best_metric():
    def func(metric, sparsity):
        return metric
    return func