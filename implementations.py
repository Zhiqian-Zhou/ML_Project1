"""EPFL CS-433 Machine Learning - Project 1.

Canonical implementations of the six core regression and classification algorithms.
All methods operate on 1D target vectors y and 2D design matrices tx.
"""

import numpy as np

# ==============================================================================
# Helper Functions (Losses, Gradients, and Iterators)
# ==============================================================================


def compute_mse_loss(y, tx, w):
    """Compute mean squared error loss with a factor of 0.5.

    Loss formula: L(w) = (1 / (2 * N)) * ||y - tx @ w||^2
    """
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Compute gradient of MSE loss with respect to parameter vector w.

    Gradient formula: grad = -(1 / N) * tx.T @ (y - tx @ w)
    """
    e = y - tx @ w
    return -(tx.T @ e) / len(y)


def sigmoid(t):
    """Numerically stable sigmoid function."""
    t_clipped = np.clip(t, -250, 250)
    return 1.0 / (1.0 + np.exp(-t_clipped))


def compute_logistic_loss(y, tx, w):
    """Compute negative log-likelihood loss for logistic regression.

    Loss formula: L(w) = -(1 / N) * sum(y * log(p) + (1 - y) * log(1 - p))
    Evaluated stably via logaddexp.
    """
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Compute gradient of negative log-likelihood with respect to w.

    Gradient formula: grad = (1 / N) * tx.T @ (sigmoid(tx @ w) - y)
    """
    pred = sigmoid(tx @ w)
    return (tx.T @ (pred - y)) / len(y)


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """Generate a mini-batch iterator for a dataset.

    Canonical sampling logic adapted from course lab 2.
    """
    data_size = len(y)
    batch_size = min(data_size, batch_size)
    max_batches = int(data_size / batch_size)
    remainder = data_size - max_batches * batch_size

    if shuffle:
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start
        end_index = start_index + batch_size
        yield y[start_index:end_index], tx[start_index:end_index]


# ==============================================================================
# 6 Canonical ML Methods (Table 1)
# ==============================================================================


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: shape=(N,) target vector.
        tx: shape=(N, D) design matrix.
        initial_w: shape=(D,) initial weight vector.
        max_iters: int, number of gradient descent steps.
        gamma: float, step size / learning rate.

    Returns:
        (w, loss): final weight vector w of shape (D,) and final MSE loss.
    """
    w = np.array(initial_w, dtype=np.float64, copy=True)
    for _ in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size = 1).

    Args:
        y: shape=(N,) target vector.
        tx: shape=(N, D) design matrix.
        initial_w: shape=(D,) initial weight vector.
        max_iters: int, number of SGD steps.
        gamma: float, step size / learning rate.

    Returns:
        (w, loss): final weight vector w of shape (D,) and final MSE loss on full dataset.
    """
    w = np.array(initial_w, dtype=np.float64, copy=True)
    for _ in range(max_iters):
        for y_b, tx_b in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_mse_gradient(y_b, tx_b, w)
            w = w - gamma * grad
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations.

    Args:
        y: shape=(N,) target vector.
        tx: shape=(N, D) design matrix.

    Returns:
        (w, loss): optimal weight vector w of shape (D,) and MSE loss.
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Args:
        y: shape=(N,) target vector.
        tx: shape=(N, D) design matrix.
        lambda_: float, regularization parameter.

    Returns:
        (w, loss): optimal weight vector w of shape (D,) and unpenalized MSE loss.
    """
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent for binary labels in {0, 1}.

    Args:
        y: shape=(N,) binary target vector in {0, 1}.
        tx: shape=(N, D) design matrix.
        initial_w: shape=(D,) initial weight vector.
        max_iters: int, number of gradient descent steps.
        gamma: float, step size / learning rate.

    Returns:
        (w, loss): final weight vector w of shape (D,) and final negative log-likelihood loss.
    """
    w = np.array(initial_w, dtype=np.float64, copy=True)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent for binary labels in {0, 1}.

    Args:
        y: shape=(N,) binary target vector in {0, 1}.
        tx: shape=(N, D) design matrix.
        lambda_: float, regularization parameter.
        initial_w: shape=(D,) initial weight vector.
        max_iters: int, number of gradient descent steps.
        gamma: float, step size / learning rate.

    Returns:
        (w, loss): final weight vector w of shape (D,) and unpenalized negative log-likelihood loss.
    """
    w = np.array(initial_w, dtype=np.float64, copy=True)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = compute_logistic_loss(y, tx, w)
    return w, loss
