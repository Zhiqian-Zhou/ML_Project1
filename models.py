"""Optional class-imbalance tools: weighted regularized logistic regression and undersampling.

These are ablation switches only (DESIGN section 2): the default model is
implementations.reg_logistic_regression with a decision threshold tuned on the
training scores. implementations.py is not modified.
"""

import numpy as np

from implementations import sigmoid


def weighted_reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma, pos_weight
):
    """Regularized logistic regression with a weight on the positive class (GD).

    Sample weights s_i = pos_weight if y_i == 1 else 1, normalized to mean 1, so
    the loss keeps the scale of the unweighted one. With pos_weight == 1 this is
    exactly implementations.reg_logistic_regression.

    Loss formula: L(w) = (1 / N) * sum(s_i * (log(1 + exp(z_i)) - y_i * z_i)), z = tx @ w
    Gradient: grad = (1 / N) * tx.T @ (s * (sigmoid(tx @ w) - y)) + 2 * lambda_ * w

    Args:
        y: shape=(N,) binary target vector in {0, 1}.
        tx: shape=(N, D) design matrix.
        lambda_: float, regularization parameter.
        initial_w: shape=(D,) initial weight vector.
        max_iters: int, number of gradient descent steps.
        gamma: float, step size / learning rate.
        pos_weight: float > 0, weight of the positive class (None means 1).

    Returns:
        (w, loss): final weight vector w of shape (D,) and unpenalized weighted
        negative log-likelihood loss (numpy scalar).
    """
    if pos_weight is None:
        pos_weight = 1.0
    if pos_weight <= 0:
        raise ValueError("pos_weight must be > 0")
    y = np.asarray(y, dtype=np.float64)
    s = np.where(y == 1, float(pos_weight), 1.0)
    s = s / np.mean(s)
    w = np.array(initial_w, dtype=np.float64, copy=True)
    for _ in range(max_iters):
        pred = sigmoid(tx @ w)
        grad = (tx.T @ (s * (pred - y))) / len(y) + 2 * lambda_ * w
        w = w - gamma * grad
    z = tx @ w
    loss = np.mean(s * (np.logaddexp(0, z) - y * z))
    return w, loss


def undersample_indices(y, ratio, seed):
    """Row indices of a class-balanced subsample: all positives plus random negatives.

    Use it only on the rows that train the model (inside a training fold).

    Args:
        y: shape=(N,) labels; positives are y == 1 (works for {0, 1} and {-1, 1}).
        ratio: float > 0, number of negatives kept per positive.
        seed: int, seed of np.random.default_rng.

    Returns:
        Sorted int array with all positive rows and min(n_neg, round(ratio * n_pos))
        negative rows drawn without replacement.
    """
    y = np.asarray(y)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y != 1)
    n_keep = min(neg.size, int(round(ratio * pos.size)))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(neg, size=n_keep, replace=False)
    return np.sort(np.concatenate([pos, chosen]).astype(int))
