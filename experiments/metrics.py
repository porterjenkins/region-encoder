import numpy as np



def root_mean_squared_error(y_true, y_hat):

    squared_errs = np.power(y_hat - y_true, 2)
    return np.sqrt(np.mean(squared_errs))


def mean_relative_error(y_true, y_hat):

    abs_err = np.abs(y_hat - y_true)
    relative_abs_err = abs_err / y_true

    return np.mean(relative_abs_err)

