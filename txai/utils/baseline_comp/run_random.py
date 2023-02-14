import torch

def run_random(
        model = None,
        X = None,
        time_input = None,
        y = None,
        device = None,):

    # Really basic - just get a random explanation
    return torch.randn_like(X).squeeze() if X is not None else 0


def screen_random(
        model,
        test_tuples, 
        only_correct = True,
        device = None):
    '''
    Screens over an entire test set to produce explanations for random Explainer

    - Assumes all input tensors are on same device

    test_tuples: list of tuples
        - [(X_0, time_0, y_0), ..., (X_N, time_N, y_N)]
    '''

    out_exp = []

    model.eval()
    for X, time, y in test_tuples:

        exp = run_random(X)

        out_exp.append(exp)

    return out_exp