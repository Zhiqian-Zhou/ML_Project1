"""The 6 required ML methods of CS-433 Project 1 (auto-graded).

STATUS: template. Every function below raises NotImplementedError until it is
implemented. Do not rename functions or change argument order: the graders
import them by name and call them positionally.

Conventions checked by the public tests (grading_tests/test_project1_public.py):
    * Shapes: y (N,), tx (N, D), w (D,). Always 1-D vectors, never (N, 1).
    * Every method returns (w, loss): the LAST weight vector and the loss
      computed AT that last w (i.e. after the final update).
    * loss must be a numpy scalar (loss.ndim == 0). Do NOT call .item(),
      return e.g. np.mean(...) directly.
    * max_iters == 0 must return (initial_w, loss(initial_w)).
    * Do not modify initial_w in place: use w = w - gamma * grad.
    * MSE = 1 / (2N) * sum(e ** 2), with e = y - tx @ w.
    * Logistic regression uses labels y in {0, 1}.
    * Regularization term is lambda_ * ||w||^2 (gradient: + 2 * lambda_ * w).
    * Returned loss of regularized methods does NOT include the penalty.
    * Only numpy. np.linalg.lstsq is forbidden.

Reference code from the labs (adapt, do not copy blindly):
    labs/ex02/solution/gradient_descent.py, stochastic_gradient_descent.py
    labs/ex03/solution/ex03.ipynb (least_squares, ridge_regression)
    labs/ex05/solution/ex05.ipynb (sigmoid, logistic loss and gradient)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Loss and gradient helpers
# ---------------------------------------------------------------------------


def compute_mse(y, tx, w):
    """Compute the MSE loss 1/(2N) * ||y - tx w||^2.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D).
        w: numpy array of shape (D,).

    Returns:
        Numpy scalar, the MSE loss.
    """
    raise NotImplementedError("Pending: compute_mse")


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss: -1/N * tx^T (y - tx w).

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D).
        w: numpy array of shape (D,).

    Returns:
        Numpy array of shape (D,).
    """
    raise NotImplementedError("Pending: compute_mse_gradient")


def sigmoid(t):
    """Apply the logistic function 1 / (1 + exp(-t)) element-wise.

    Should be numerically stable for large |t| (no overflow warnings).

    Args:
        t: numpy array or scalar.

    Returns:
        Numpy array of the same shape as t, with values in (0, 1).
    """
    raise NotImplementedError("Pending: sigmoid")


def compute_logistic_loss(y, tx, w):
    """Compute the mean negative log-likelihood for labels in {0, 1}.

    loss = mean(log(1 + exp(tx w)) - y * (tx w)).
    Hint: np.logaddexp(0, z) computes log(1 + exp(z)) stably.

    Args:
        y: numpy array of shape (N,) with values in {0, 1}.
        tx: numpy array of shape (N, D).
        w: numpy array of shape (D,).

    Returns:
        Numpy scalar, the logistic loss.
    """
    raise NotImplementedError("Pending: compute_logistic_loss")


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic loss: 1/N * tx^T (sigmoid(tx w) - y).

    Args:
        y: numpy array of shape (N,) with values in {0, 1}.
        tx: numpy array of shape (N, D).
        w: numpy array of shape (D,).

    Returns:
        Numpy array of shape (D,).
    """
    raise NotImplementedError("Pending: compute_logistic_gradient")


# ---------------------------------------------------------------------------
# The 6 required methods (Table 1 of the project description)
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full gradient descent on the MSE loss.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D).
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: int, number of gradient steps.
        gamma: float, step size.

    Returns:
        w: numpy array of shape (D,), the last weights.
        loss: numpy scalar, MSE at the last weights.
    """
    raise NotImplementedError("Pending: mean_squared_error_gd")


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    At each iteration, sample one random index n and step along the
    gradient of 1/2 * (y_n - x_n^T w)^2. The returned loss is the MSE on the
    full (y, tx), not on the last sample.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D).
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: int, number of SGD steps.
        gamma: float, step size.

    Returns:
        w: numpy array of shape (D,), the last weights.
        loss: numpy scalar, MSE on the full data at the last weights.
    """
    raise NotImplementedError("Pending: mean_squared_error_sgd")


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Solve tx^T tx w = tx^T y with np.linalg.solve (never invert the matrix,
    np.linalg.lstsq is not allowed).

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D).

    Returns:
        w: numpy array of shape (D,), optimal weights.
        loss: numpy scalar, MSE at w.
    """
    raise NotImplementedError("Pending: least_squares")


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Minimizes 1/(2N) ||y - tx w||^2 + lambda_ ||w||^2, i.e. solves
    (tx^T tx + 2 N lambda_ I) w = tx^T y.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D).
        lambda_: float, regularization strength.

    Returns:
        w: numpy array of shape (D,), optimal weights.
        loss: numpy scalar, MSE at w WITHOUT the penalty term.
    """
    raise NotImplementedError("Pending: ridge_regression")


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (labels in {0, 1}).

    Args:
        y: numpy array of shape (N,) with values in {0, 1}.
        tx: numpy array of shape (N, D).
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: int, number of gradient steps.
        gamma: float, step size.

    Returns:
        w: numpy array of shape (D,), the last weights.
        loss: numpy scalar, logistic loss at the last weights.
    """
    raise NotImplementedError("Pending: logistic_regression")


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Penalty lambda_ * ||w||^2, so the gradient is the logistic gradient
    plus 2 * lambda_ * w. Note the argument order: lambda_ before initial_w.

    Args:
        y: numpy array of shape (N,) with values in {0, 1}.
        tx: numpy array of shape (N, D).
        lambda_: float, regularization strength.
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: int, number of gradient steps.
        gamma: float, step size.

    Returns:
        w: numpy array of shape (D,), the last weights.
        loss: numpy scalar, logistic loss at the last weights WITHOUT the
            penalty term.
    """
    raise NotImplementedError("Pending: reg_logistic_regression")
