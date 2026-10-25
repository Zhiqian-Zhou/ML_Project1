"""Small fully connected neural network (multi-layer perceptron), numpy only.

Link with the course (lecture on neural networks):
    * Model: x -> [Linear -> ReLU] x L -> Linear -> logit; the probability
      is sigmoid(logit), as in logistic regression.
    * Loss: binary cross-entropy (the logistic loss) + L2 weight decay
      lambda * ||W||^2 on the weight matrices (like ridge; biases are not
      penalized).
    * Gradients by backpropagation (chain rule, layer by layer), mini-batch
      updates with the Adam optimizer (SGD with adaptive step sizes).
    * Early stopping on an inner stratified validation split, another
      regularization; the logits of several networks (different seeds) are
      averaged to reduce the variance.
"""

import numpy as np

DEFAULT_PARAMS = {
    "hidden": [64],  # sizes of the hidden layers
    "lr": 1e-3,  # Adam step size
    "batch_size": 512,
    "max_epochs": 40,
    "patience": 4,  # epochs without validation improvement before stopping
    "weight_decay": 1e-3,  # L2 penalty lambda * ||W||^2
    "val_frac": 0.2,  # inner validation share of the training rows
    "seed": 0,
    "n_seeds": 10,  # average the logits of n_seeds networks (seed, seed+1, ...)
}


def sigmoid(t):
    """Numerically stable logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def bce_from_logits(y01, logits):
    """Mean binary cross-entropy computed from logits (stable form)."""
    return np.mean(np.logaddexp(0.0, logits) - y01 * logits)


def stratified_split(y, frac, rng):
    """Return (train_idx, val_idx) with the same positive share in both parts."""
    val = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        val.append(idx[: int(round(frac * len(idx)))])
    val = np.sort(np.concatenate(val))
    mask = np.ones(len(y), dtype=bool)
    mask[val] = False
    return np.flatnonzero(mask), val


def init_params(sizes, rng):
    """He initialization of the weights, zero biases.

    Args:
        sizes: list of layer widths [D, H1, ..., 1].
        rng: numpy random Generator.

    Returns:
        list of [W, b] pairs, W of shape (d_in, d_out), float32.
    """
    layers = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        W = rng.standard_normal((d_in, d_out)) * np.sqrt(2.0 / d_in)
        layers.append([W.astype(np.float32), np.zeros(d_out, dtype=np.float32)])
    return layers


def forward(layers, x):
    """Forward pass.

    Args:
        layers: list of [W, b].
        x: inputs of shape (N, D).

    Returns:
        logits: shape (N,).
        cache: list of (input of the layer, pre-activation) for backward.
    """
    cache = []
    a = x
    for W, b in layers[:-1]:
        z = a @ W + b
        cache.append((a, z))
        a = np.maximum(z, 0.0)  # ReLU
    W, b = layers[-1]
    cache.append((a, None))
    return (a @ W + b)[:, 0], cache


def backward(layers, cache, y01, logits, weight_decay):
    """Backpropagation of the mean BCE loss + weight decay.

    Returns:
        list of [dW, db] with the same shapes as layers.
    """
    n = len(y01)
    # derivative of the BCE w.r.t. the logit: sigmoid(logit) - y
    delta = ((sigmoid(logits) - y01) / n)[:, None].astype(np.float32)
    grads = [None] * len(layers)
    for i in range(len(layers) - 1, -1, -1):
        W, _ = layers[i]
        a_in = cache[i][0]
        grads[i] = [a_in.T @ delta + 2 * weight_decay * W, delta.sum(axis=0)]
        if i > 0:
            # chain rule: back through the weights, then the ReLU
            delta = (delta @ W.T) * (cache[i - 1][1] > 0)
    return grads


def adam_step(layers, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """One Adam update of all weights and biases (in place).

    m and v are running means of the gradient and of its square; t is the
    number of the step (from 1), used for the bias correction.
    """
    for layer, g, mi, vi in zip(layers, grads, m, v):
        for j in range(2):  # j = 0: weights, j = 1: biases
            mi[j] = beta1 * mi[j] + (1 - beta1) * g[j]
            vi[j] = beta2 * vi[j] + (1 - beta2) * g[j] ** 2
            m_hat = mi[j] / (1 - beta1**t)
            v_hat = vi[j] / (1 - beta2**t)
            layer[j] -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)


def predict_logits(layers, x, batch=65536):
    """Logits for x, computed by chunks to limit memory."""
    out = [forward(layers, x[i : i + batch])[0] for i in range(0, len(x), batch)]
    return np.concatenate(out)


def train_one(y01, x, x_val, y_val01, p, seed):
    """Train one network with Adam and early stopping on the validation loss.

    Returns:
        the layers with the lowest validation loss.
    """
    rng = np.random.default_rng(seed)
    layers = init_params([x.shape[1]] + list(p["hidden"]) + [1], rng)
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    best_loss, best_epoch, t = np.inf, 0, 0
    best_layers = [[W.copy(), b.copy()] for W, b in layers]
    n = len(y01)
    for epoch in range(p["max_epochs"]):
        order = rng.permutation(n)
        for start in range(0, n, p["batch_size"]):
            idx = order[start : start + p["batch_size"]]
            logits, cache = forward(layers, x[idx])
            grads = backward(layers, cache, y01[idx], logits, p["weight_decay"])
            t += 1
            adam_step(layers, grads, m, v, t, p["lr"])
        # early stopping: keep the best epoch, stop after `patience` worse ones
        val_loss = bce_from_logits(y_val01, predict_logits(layers, x_val))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_layers = [[W.copy(), b.copy()] for W, b in layers]
        elif epoch - best_epoch >= p["patience"]:
            break
    return best_layers


def train(y, tx, params=None):
    """Fit n_seeds networks on the same inner train / validation split.

    Args:
        y: labels in {-1, 1}, shape (N,).
        tx: features, shape (N, D) (preprocessing.transform output).
        params: dict, missing keys come from DEFAULT_PARAMS.

    Returns:
        model: dict with the list of networks.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    tx = np.asarray(tx, dtype=np.float32)
    rng = np.random.default_rng(p["seed"])
    tr, va = stratified_split(y, p["val_frac"], rng)
    y01 = (y == 1).astype(np.float32)
    nets = [
        train_one(y01[tr], tx[tr], tx[va], y01[va], p, p["seed"] + s)
        for s in range(p["n_seeds"])
    ]
    return {"nets": nets}


def score(model, tx):
    """Mean logit of the networks (higher = more likely positive)."""
    tx = np.asarray(tx, dtype=np.float32)
    return np.mean([predict_logits(net, tx) for net in model["nets"]], axis=0)
