import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils.validation import check_consistent_length

def mase_metric(
    predictions,
    references,
    training,
    periodicity=1,
    sample_weight=None,
    multioutput="uniform_average",
):
    y_pred_naive = training[:-periodicity]
    mae_naive = mean_absolute_error(training[periodicity:], y_pred_naive, multioutput=multioutput)

    mae_score = mean_absolute_error(
        references,
        predictions,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )

    epsilon = np.finfo(np.float64).eps
    mase_score = mae_score / np.maximum(mae_naive, epsilon)

    return {"mase": mase_score}

def symmetric_mean_absolute_percentage_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Symmetric Mean absolute percentage error (sMAPE) metric using sklearn's api and helpers.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute percentage error
        is returned for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.
        sMAPE output is non-negative floating point. The best value is 0.0.
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    epsilon = np.finfo(np.float64).eps
    smape = 2 * np.abs(y_pred - y_true) / (np.maximum(np.abs(y_true), epsilon) + np.maximum(np.abs(y_pred), epsilon))
    output_errors = np.average(smape, weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def smape_metric(predictions, references, sample_weight=None, multioutput="uniform_average"):
    smape_score = symmetric_mean_absolute_percentage_error(
        references,
        predictions,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )
    return {"smape": smape_score}