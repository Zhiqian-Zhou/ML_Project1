"""Pure NumPy implementations of GBDT, Multi-Seed MLP, and the Champion Ensemble.

Provides:
    1. GradientBoostedTrees: Histogram-based GBDT with Newton leaf updates and early stopping.
    2. MLPEnsemble: Multi-seed neural network ensemble with Adam optimization and early stopping.
    3. CardiacEnsemble: End-to-end composite model combining GBDT and MLP with clinical prior boosting.
    4. ModelSerializer: Safe, portable pure-NumPy model serialization (allow_pickle=False).
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# ==============================================================================
# Model Serialization (Safe pure-NumPy archive without pickle)
# ==============================================================================


class ModelSerializer:
    """Portable NumPy model serializer using JSON manifests and pure ndarrays."""

    @staticmethod
    def save(path, model_dict):
        """Save nested dicts/lists and numeric ndarrays into a compressed .npz archive."""
        arrays = {}

        def encode(obj):
            if isinstance(obj, np.ndarray):
                key = f"arr_{len(arrays)}"
                arrays[key] = np.ascontiguousarray(obj)
                return {"__ndarray__": key}
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, tuple):
                return {"__tuple__": [encode(v) for v in obj]}
            if isinstance(obj, list):
                return [encode(v) for v in obj]
            if isinstance(obj, dict):
                return {"__dict__": [[encode(k), encode(v)] for k, v in obj.items()]}
            return obj

        manifest = encode(model_dict)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, __manifest__=np.array(json.dumps(manifest)), **arrays)

    @staticmethod
    def load(path):
        """Load model dictionary from a compressed .npz archive with allow_pickle=False."""
        with np.load(path, allow_pickle=False) as archive:

            def decode(obj):
                if isinstance(obj, list):
                    return [decode(v) for v in obj]
                if isinstance(obj, dict):
                    if "__ndarray__" in obj:
                        return archive[obj["__ndarray__"]].copy()
                    if "__tuple__" in obj:
                        return tuple(decode(v) for v in obj["__tuple__"])
                    if "__dict__" in obj:
                        return {decode(k): decode(v) for k, v in obj["__dict__"]}
                return obj

            manifest_str = str(archive["__manifest__"])
            return decode(json.loads(manifest_str))


save_model = ModelSerializer.save
load_model = ModelSerializer.load


# ==============================================================================
# Mathematical Utilities
# ==============================================================================


def sigmoid(t):
    """Numerically stable logistic sigmoid function."""
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def logistic_loss(y01, f, w=None):
    """Mean logistic loss of logits f for labels y01 in {0, 1}."""
    loss_elements = np.logaddexp(0.0, f) - y01 * f
    if w is None:
        return float(np.mean(loss_elements))
    return float(np.sum(w * loss_elements) / np.sum(w))


def stratified_split(y, frac, seed_or_rng):
    """Indices (train, valid) with a stratified share `frac` of each class in valid."""
    if isinstance(seed_or_rng, np.random.Generator):
        rng = seed_or_rng
    else:
        rng = np.random.default_rng(seed_or_rng)
    va = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        va.append(idx[: int(round(frac * len(idx)))])
    va = np.sort(np.concatenate(va))
    tr = np.setdiff1d(np.arange(len(y)), va)
    return tr, va


# ==============================================================================
# Gradient Boosted Decision Trees (Histogram-based, Pure NumPy)
# ==============================================================================

DEFAULT_GBM_PARAMS = {
    "refit_full": False,
    "n_trees": 1500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_leaf": 50,
    "min_hessian": 1.0,
    "l2": 5.0,
    "min_gain": 0.0,
    "n_bins": 64,
    "subsample": 0.7,
    "colsample": 0.8,
    "valid_frac": 0.2,
    "patience": 100,
    "target_enc_last": 100,
    "seed": 42,
    "w_pos": 1.0,
}

LEAKY_COLUMNS = [
    "HAREHAB1",
    "CVDASPRN",
    "ASPUNSAF",
    "RLIVPAIN",
    "RDUCHART",
    "RDUCSTRK",
    "STREHAB1",
]


def tree_features(xc, names, x_leaky=None):
    """Prepare design matrix for trees: drops leaky columns, keeps _STATE last."""
    assert names[-1] == "_STATE"
    keep = np.array([n not in LEAKY_COLUMNS for n in names])
    x = np.asarray(xc)[:, keep]
    if x_leaky is None:
        return x
    return np.hstack([x[:, :-1], x_leaky, x[:, -1:]])


def fit_target_encoding(codes, y01, m):
    """Smoothed prevalence of each category (e.g., state code)."""
    prior = float(np.mean(y01))
    cats, inv = np.unique(codes, return_inverse=True)
    pos = np.bincount(inv, weights=y01)
    cnt = np.bincount(inv)
    return cats, (pos + m * prior) / (cnt + m), prior


def apply_target_encoding(tx, encoding):
    """Replace the last column (category codes) with smoothed target risk values."""
    cats, values, prior = encoding
    codes = tx[:, -1]
    pos = np.clip(np.searchsorted(cats, codes), 0, len(cats) - 1)
    enc = np.where(cats[pos] == codes, values[pos], prior)
    return np.column_stack([tx[:, :-1], enc])


def fit_bins(x, n_bins):
    """Fit quantile bin edges per feature on non-NaN values."""
    edges = []
    for j in range(x.shape[1]):
        col = x[:, j]
        values = np.unique(col[~np.isnan(col)])
        if len(values) <= n_bins:
            edges.append((values[:-1] + values[1:]) / 2)
        else:
            qs = np.linspace(0, 1, n_bins + 1)[1:-1]
            edges.append(np.unique(np.quantile(col[~np.isnan(col)], qs)))
    return edges


def apply_bins(x, edges, n_bins):
    """Map continuous features into uint8 bin indices; NaN assigned to bin n_bins."""
    xb = np.empty(x.shape, dtype=np.uint8)
    for j, e in enumerate(edges):
        col = x[:, j]
        xb[:, j] = np.searchsorted(e, col, side="right")
        xb[np.isnan(col), j] = n_bins
    return xb


def build_histograms(xb, g, h, rows, slot, n_slots, n_bins):
    """Accumulate gradient and hessian histograms per (node, feature, bin)."""
    d, nb = xb.shape[1], n_bins + 1
    size = n_slots * nb
    G, H, C = [np.empty((d, n_slots, nb)) for _ in range(3)]
    g_r, h_r = g[rows], h[rows]
    sub = np.ascontiguousarray(xb[rows].T)
    base = slot * nb
    for j in range(d):
        idx = base + sub[j]
        G[j] = np.bincount(idx, g_r, size).reshape(n_slots, nb)
        H[j] = np.bincount(idx, h_r, size).reshape(n_slots, nb)
        C[j] = np.bincount(idx, minlength=size).reshape(n_slots, nb)
    return [A.transpose(1, 0, 2) for A in (G, H, C)]


def best_splits(G, H, C, col_mask, p):
    """Find the optimal split point per node maximizing second-order gain."""
    l2, nv = p["l2"], G.shape[2] - 1
    stats = []
    for A in (G, H, C):
        cum = np.cumsum(A[:, :, :nv], axis=2)
        nan = A[:, :, nv:]
        stats.append(np.stack([cum, cum + nan], axis=3))
    GL, HL, CL = stats
    Gt, Ht, Ct = [A[:, 0, :].sum(axis=1)[:, None, None, None] for A in (G, H, C)]
    GR, HR, CR = Gt - GL, Ht - HL, Ct - CL
    gain = GL**2 / (HL + l2) + GR**2 / (HR + l2) - Gt**2 / (Ht + l2)
    ok = (
        (CL >= p["min_leaf"])
        & (CR >= p["min_leaf"])
        & (HL >= p["min_hessian"])
        & (HR >= p["min_hessian"])
        & col_mask[None, :, None, None]
    )
    gain = np.where(ok, gain, -np.inf)
    n_slots = G.shape[0]
    best = np.argmax(gain.reshape(n_slots, -1), axis=1)
    feat, thr, nan_left = np.unravel_index(best, gain.shape[1:])
    pick = (np.arange(n_slots), feat, thr, nan_left)
    return feat, thr, nan_left, gain[pick], CL[pick]


def child_histograms(xb, g, h, rows, slot, hists, par, child_slot, n_left, n_bins):
    """Compute child histograms using the histogram subtraction trick."""
    n_next = 2 * len(par)
    n_tot = hists[2][:, 0, :].sum(axis=1)
    small_is_left = n_left[par] <= n_tot[par] - n_left[par]
    small = np.where(small_is_left, child_slot[par, 0], child_slot[par, 1])
    large = np.where(small_is_left, child_slot[par, 1], child_slot[par, 0])
    is_small = np.zeros(n_next, dtype=bool)
    is_small[small] = True
    m = is_small[slot]
    new = [np.zeros((n_next,) + hists[0].shape[1:]) for _ in range(3)]
    hs = build_histograms(xb, g, h, rows[m], slot[m], n_next, n_bins)
    for A_new, A_small, A_par in zip(new, hs, hists):
        A_new[small] = A_small[small]
        A_new[large] = A_par[par] - A_small[small]
    return new


def build_tree(xb, g, h, rows, col_mask, p):
    """Grow one decision tree level by level with Newton leaf steps."""
    nbin, lr = p["n_bins"], p["learning_rate"]
    keys = ("feat", "thr", "nan_left", "left", "right", "value")
    tree = {k: [] for k in keys}

    def new_node():
        for k, v in zip(keys, (-1, 0, 0, 0, 0, 0.0)):
            tree[k].append(v)
        return len(tree["feat"]) - 1

    level = [new_node()]
    slot = np.zeros(len(rows), dtype=np.int64)
    G, H, C = build_histograms(xb, g, h, rows, slot, 1, nbin)
    for depth in range(p["max_depth"] + 1):
        Gt, Ht = G[:, 0, :].sum(axis=1), H[:, 0, :].sum(axis=1)
        split = np.zeros(len(level), dtype=bool)
        if depth < p["max_depth"]:
            f, b, nl, gain, n_left = best_splits(G, H, C, col_mask, p)
            split = gain > p["min_gain"]
        for s in np.where(~split)[0]:
            tree["value"][level[s]] = -lr * Gt[s] / (Ht[s] + p["l2"])
        if not split.any():
            break
        par = np.where(split)[0]
        next_level, child_slot = [], -np.ones((len(level), 2), dtype=np.int64)
        for s in par:
            node = level[s]
            tree["feat"][node], tree["thr"][node] = int(f[s]), int(b[s])
            tree["nan_left"][node] = int(nl[s])
            tree["left"][node], tree["right"][node] = new_node(), new_node()
            child_slot[s] = [len(next_level), len(next_level) + 1]
            next_level += [tree["left"][node], tree["right"][node]]
        keep = split[slot]
        rows, slot = rows[keep], slot[keep]
        v = xb[rows, f[slot]]
        go_left = np.where(v == nbin, nl[slot] == 1, v <= b[slot])
        slot = np.where(go_left, child_slot[slot, 0], child_slot[slot, 1])
        G, H, C = child_histograms(
            xb, g, h, rows, slot, (G, H, C), par, child_slot, n_left, nbin
        )
        level = next_level
    return {k: np.array(v) for k, v in tree.items()}


def predict_tree(tree, xb, n_bins):
    """Vectorized inference of a single tree on binned features xb."""
    node = np.zeros(len(xb), dtype=np.int64)
    rows = np.arange(len(xb))
    while True:
        f = tree["feat"][node]
        inner = f >= 0
        if not inner.any():
            return tree["value"][node]
        v = xb[rows, np.maximum(f, 0)]
        go_left = np.where(
            v == n_bins, tree["nan_left"][node] == 1, v <= tree["thr"][node]
        )
        child = np.where(go_left, tree["left"][node], tree["right"][node])
        node = np.where(inner, child, node)


def boost(xb, y01, xb_val, y_val, p):
    """Execute boosting rounds with validation-based early stopping."""
    rng = np.random.default_rng(p["seed"])
    n, d = xb.shape
    w_pos = float(p.get("w_pos", 1.0))
    w_train = np.where(y01 == 1, w_pos, 1.0)
    early_stopping = xb_val is not None
    w_val = np.where(y_val == 1, w_pos, 1.0) if early_stopping else None

    prior = np.clip(np.sum(w_train * y01) / np.sum(w_train), 1e-6, 1 - 1e-6)
    f0 = float(np.log(prior / (1 - prior)))
    f = np.full(n, f0)
    f_val = np.full(len(xb_val), f0) if early_stopping else None
    trees, val_curve = [], []
    for t in range(p["n_trees"]):
        prob = sigmoid(f)
        g = w_train * (prob - y01)
        h = np.maximum(w_train * prob * (1.0 - prob), 1e-12)
        rows = np.arange(n)
        if p["subsample"] < 1:
            rows = np.sort(rng.choice(n, int(p["subsample"] * n), replace=False))
        col_mask = np.ones(d, dtype=bool)
        if p["colsample"] < 1:
            col_mask[:] = False
            n_cols = max(1, int(p["colsample"] * d))
            col_mask[rng.choice(d, n_cols, replace=False)] = True
        tree = build_tree(xb, g, h, rows, col_mask, p)
        trees.append(tree)
        f += predict_tree(tree, xb, p["n_bins"])
        if not early_stopping:
            continue
        f_val += predict_tree(tree, xb_val, p["n_bins"])
        val_curve.append(logistic_loss(y_val, f_val, w_val))
        if t - int(np.argmin(val_curve)) >= p["patience"]:
            break
    return (
        f0,
        trees[: int(np.argmin(val_curve)) + 1] if early_stopping else trees,
    )


class GradientBoostedTrees:
    """Histogram-based Gradient Boosted Trees for binary classification (NumPy only)."""

    def __init__(self, **params):
        self.params = dict(DEFAULT_GBM_PARAMS, **params)
        self.encoding_ = None
        self.edges_ = None
        self.f0_ = 0.0
        self.trees_ = []
        self.selected_rounds_ = 0
        self.training_rows_ = 0

    def fit(self, tx, y):
        """Fit model on features tx (last column _STATE) and labels y in {-1, 1}."""
        p = self.params
        y01 = (y == 1).astype(np.float64)
        tr, va = stratified_split(y01, p["valid_frac"], p["seed"])
        self.encoding_ = fit_target_encoding(tx[tr, -1], y01[tr], p["target_enc_last"])
        raw_tx = tx
        tx_enc = apply_target_encoding(tx, self.encoding_)
        self.edges_ = fit_bins(tx_enc[tr] if p["refit_full"] else tx_enc, p["n_bins"])
        xb = apply_bins(tx_enc, self.edges_, p["n_bins"])
        f0, trees = boost(xb[tr], y01[tr], xb[va], y01[va], p)
        self.selected_rounds_ = len(trees)

        if p["refit_full"]:
            self.encoding_ = fit_target_encoding(
                raw_tx[:, -1], y01, p["target_enc_last"]
            )
            tx_enc = apply_target_encoding(raw_tx, self.encoding_)
            self.edges_ = fit_bins(tx_enc, p["n_bins"])
            xb = apply_bins(tx_enc, self.edges_, p["n_bins"])
            f0, trees = boost(
                xb, y01, None, None, dict(p, n_trees=self.selected_rounds_)
            )

        self.f0_ = f0
        self.trees_ = trees
        self.training_rows_ = len(y) if p["refit_full"] else len(tr)
        return self

    def predict_logits(self, tx):
        """Compute continuous log-odds predictions for tx."""
        tx_enc = apply_target_encoding(tx, self.encoding_)
        xb = apply_bins(tx_enc, self.edges_, self.params["n_bins"])
        f = np.full(len(tx), self.f0_)
        for tree in self.trees_:
            f += predict_tree(tree, xb, self.params["n_bins"])
        return f

    def predict_proba(self, tx):
        """Compute positive class probabilities."""
        return sigmoid(self.predict_logits(tx))

    def to_dict(self):
        """Export model parameters and trees as a checkpoint dictionary."""
        return {
            "encoding": self.encoding_,
            "edges": self.edges_,
            "n_bins": self.params["n_bins"],
            "selected_rounds": self.selected_rounds_,
            "training_rows": self.training_rows_,
            "refit_full": self.params["refit_full"],
            "f0": self.f0_,
            "trees": self.trees_,
        }

    @classmethod
    def from_dict(cls, d, **params):
        """Instantiate a fitted model from a checkpoint dictionary."""
        model = cls(n_bins=d["n_bins"], refit_full=d["refit_full"], **params)
        model.encoding_ = d["encoding"]
        model.edges_ = d["edges"]
        model.f0_ = d["f0"]
        model.trees_ = d["trees"]
        model.selected_rounds_ = d["selected_rounds"]
        model.training_rows_ = d["training_rows"]
        return model


def train_gbm(y, tx, params=None):
    """Functional wrapper for GBDT training."""
    model = GradientBoostedTrees(**(params or {}))
    model.fit(tx, y)
    d = model.to_dict()
    return d


def score_gbm(model, tx):
    """Functional wrapper for GBDT scoring."""
    if isinstance(model, GradientBoostedTrees):
        return model.predict_logits(tx)
    return GradientBoostedTrees.from_dict(model).predict_logits(tx)


# ==============================================================================
# Multi-Layer Perceptron (Neural Network Ensemble, Pure NumPy)
# ==============================================================================

DEFAULT_MLP_PARAMS = {
    "split_per_seed": False,
    "hidden": [64],
    "lr": 1e-3,
    "batch_size": 512,
    "max_epochs": 40,
    "patience": 4,
    "weight_decay": 1e-3,
    "val_frac": 0.2,
    "seed": 0,
    "n_seeds": 10,
}


def bce_from_logits(y01, logits):
    """Numerically stable mean binary cross-entropy."""
    return np.mean(np.logaddexp(0.0, logits) - y01 * logits)


def init_mlp_params(sizes, rng):
    """He normal initialization for weights and zero biases."""
    layers = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        W = rng.standard_normal((d_in, d_out)) * np.sqrt(2.0 / d_in)
        layers.append([W.astype(np.float32), np.zeros(d_out, dtype=np.float32)])
    return layers


def mlp_forward(layers, x):
    """Forward pass through linear layers with ReLU activations."""
    cache = []
    a = x
    for W, b in layers[:-1]:
        z = a @ W + b
        cache.append((a, z))
        a = np.maximum(z, 0.0)
    W, b = layers[-1]
    cache.append((a, None))
    return (a @ W + b)[:, 0], cache


def mlp_backward(layers, cache, y01, logits, weight_decay):
    """Backpropagation of binary cross-entropy with L2 weight decay."""
    n = len(y01)
    delta = ((sigmoid(logits) - y01) / n)[:, None].astype(np.float32)
    grads = [None] * len(layers)
    for i in range(len(layers) - 1, -1, -1):
        W, _ = layers[i]
        a_in = cache[i][0]
        grads[i] = [a_in.T @ delta + 2 * weight_decay * W, delta.sum(axis=0)]
        if i > 0:
            delta = (delta @ W.T) * (cache[i - 1][1] > 0)
    return grads


def adam_step(layers, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """In-place Adam optimizer update."""
    for layer, g, mi, vi in zip(layers, grads, m, v):
        for j in range(2):
            mi[j] = beta1 * mi[j] + (1 - beta1) * g[j]
            vi[j] = beta2 * vi[j] + (1 - beta2) * g[j] ** 2
            m_hat = mi[j] / (1 - beta1**t)
            v_hat = vi[j] / (1 - beta2**t)
            layer[j] -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)


def predict_mlp_logits(layers, x, batch=65536):
    """Compute logits in chunks to minimize peak memory consumption."""
    out = [mlp_forward(layers, x[i : i + batch])[0] for i in range(0, len(x), batch)]
    return np.concatenate(out)


def train_one_mlp(y01, x, x_val, y_val01, p, seed):
    """Train a single neural network with Adam and early stopping."""
    rng = np.random.default_rng(seed)
    layers = init_mlp_params([x.shape[1]] + list(p["hidden"]) + [1], rng)
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    best_loss, best_epoch, t = np.inf, 0, 0
    best_layers = [[W.copy(), b.copy()] for W, b in layers]
    n = len(y01)
    for epoch in range(p["max_epochs"]):
        order = rng.permutation(n)
        for start in range(0, n, p["batch_size"]):
            idx = order[start : start + p["batch_size"]]
            logits, cache = mlp_forward(layers, x[idx])
            grads = mlp_backward(layers, cache, y01[idx], logits, p["weight_decay"])
            t += 1
            adam_step(layers, grads, m, v, t, p["lr"])
        val_loss = bce_from_logits(y_val01, predict_mlp_logits(layers, x_val))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_layers = [[W.copy(), b.copy()] for W, b in layers]
        elif epoch - best_epoch >= p["patience"]:
            break
    return best_layers


train_one = train_one_mlp  # Alias for backward compatibility


class MLPEnsemble:
    """Multi-Seed Neural Network Ensemble for binary classification."""

    def __init__(self, **params):
        self.params = dict(DEFAULT_MLP_PARAMS, **params)
        self.nets_ = []

    def fit(self, tx, y):
        """Fit an ensemble of neural networks with diverse seeds."""
        p = self.params
        tx = np.asarray(tx, dtype=np.float32)
        rng = np.random.default_rng(p["seed"])
        tr, va = stratified_split(y, p["val_frac"], rng)
        y01 = (y == 1).astype(np.float32)
        x_tr, x_va = tx[tr], tx[va]
        y_tr01, y_va01 = y01[tr], y01[va]
        self.nets_ = []
        for s in range(p["n_seeds"]):
            if p["split_per_seed"]:
                split_rng = np.random.default_rng(p["seed"] + s)
                tr, va = stratified_split(y, p["val_frac"], split_rng)
                x_tr, x_va = tx[tr], tx[va]
                y_tr01, y_va01 = y01[tr], y01[va]
            self.nets_.append(train_one(y_tr01, x_tr, x_va, y_va01, p, p["seed"] + s))
        return self

    def predict_logits(self, tx):
        """Average logits across all networks."""
        tx = np.asarray(tx, dtype=np.float32)
        return np.mean([predict_mlp_logits(net, tx) for net in self.nets_], axis=0)

    def predict_proba(self, tx):
        """Compute positive class probabilities."""
        return sigmoid(self.predict_logits(tx))

    def to_dict(self):
        """Export networks to checkpoint dictionary."""
        return {"nets": self.nets_}

    @classmethod
    def from_dict(cls, d, **params):
        """Instantiate fitted ensemble from checkpoint dictionary."""
        model = cls(**params)
        model.nets_ = d["nets"]
        return model


def train_mlp(y, tx, params=None):
    """Functional wrapper for MLP ensemble training."""
    model = MLPEnsemble(**(params or {}))
    model.fit(tx, y)
    return model.to_dict()


def score_mlp(model, tx):
    """Functional wrapper for MLP ensemble scoring."""
    if isinstance(model, MLPEnsemble):
        return model.predict_logits(tx)
    return MLPEnsemble.from_dict(model).predict_logits(tx)


# ==============================================================================
# Champion Cardiac Risk Ensemble (Composite Model)
# ==============================================================================


class CardiacEnsemble:
    """End-to-end composite model combining GBDT and MLP with clinical priors."""

    def __init__(
        self,
        with_leakage=False,
        top_rate=None,
        gbm_weight=0.5,
        version="1.0",
    ):
        self.with_leakage = with_leakage
        from preprocessing import CLEAN_RATE, LEAKY_RATE

        self.top_rate = top_rate or (LEAKY_RATE if with_leakage else CLEAN_RATE)
        self.gbm_weight = gbm_weight
        self.version = version
        self.risk_tables = None
        self.names = None
        self.gbm_models = []
        self.mlp_state = None
        self.flags_stats = None
        self.mlp_model = None

    def fit(self, x_raw, y, raw_names, n_jobs=3):
        """Train the full ensemble pipeline from scratch."""
        from preprocessing import (
            apply_target_risk_encodings,
            fit_preprocessor,
            fit_standardizer,
            fit_target_risk_encodings,
            model_inputs,
            standardize,
            transform,
        )

        t0 = time.time()
        y = np.asarray(y)

        # 1. Feature cleaning and empirical Bayes risk tables
        xc, names, leaky = model_inputs(x_raw, raw_names, self.with_leakage)
        self.names = names
        self.risk_tables, names_enc = fit_target_risk_encodings(y, xc, names)
        xc_enc, _ = apply_target_risk_encodings(xc, names, self.risk_tables)
        tree_block = leaky[0] if leaky is not None else None
        flag_block = leaky[1] if leaky is not None else None

        # 2. Fit 3-seed GBDT ensemble in parallel
        tx_gbm = tree_features(xc_enc, names_enc, tree_block)
        seeds = [42, 43, 44]

        def _worker(s):
            m = GradientBoostedTrees(seed=s)
            m.fit(tx_gbm, y)
            return m

        with ProcessPoolExecutor(max_workers=min(n_jobs, 3)) as pool:
            self.gbm_models = list(pool.map(_worker, seeds))

        # 3. Fit neural network preprocessor and 10-seed MLP
        self.mlp_state = fit_preprocessor(
            xc_enc[:, :-1], names_enc[:-1], {"interactions": False}, y=y
        )
        tx_mlp = transform(xc_enc[:, :-1], self.mlp_state)
        if flag_block is not None:
            mean, std = fit_standardizer(flag_block)
            self.flags_stats = (mean, std)
            tx_mlp = np.hstack([tx_mlp, standardize(flag_block, mean, std)])

        self.mlp_model = MLPEnsemble(seed=0, n_seeds=10)
        self.mlp_model.fit(tx_mlp, y)
        return self

    def predict_scores(self, x_raw, raw_names):
        """Compute blended continuous log-odds scores for test respondents."""
        from preprocessing import (
            apply_target_risk_encodings,
            model_inputs,
            standardize,
            transform,
        )

        # 1. Cleaning & risk table projection
        xc, names, leaky = model_inputs(x_raw, raw_names, self.with_leakage)
        xc_enc, names_enc = apply_target_risk_encodings(xc, names, self.risk_tables)
        tree_block = leaky[0] if leaky is not None else None
        flag_block = leaky[1] if leaky is not None else None

        # 2. GBDT score (3-seed mean)
        tx_gbm = tree_features(xc_enc, names_enc, tree_block)
        s_gbm = np.mean([m.predict_logits(tx_gbm) for m in self.gbm_models], axis=0)

        # 3. MLP score (10-seed mean)
        tx_mlp = transform(xc_enc[:, :-1], self.mlp_state)
        if self.flags_stats is not None:
            mean, std = self.flags_stats
            tx_mlp = np.hstack([tx_mlp, standardize(flag_block, mean, std)])
        s_mlp = self.mlp_model.predict_logits(tx_mlp)

        # 4. Ensemble blending & clinical prior boost
        scores = self.gbm_weight * s_gbm + (1.0 - self.gbm_weight) * s_mlp
        if tree_block is not None:
            # Boost cardiac rehab attendees (positive post-event signal)
            scores[~np.isnan(tree_block[:, 0])] += 10.0
        return scores

    def predict(self, x_raw, raw_names, top_rate_override=None):
        """Generate binary labels in {-1, 1} via calibrated percentile threshold."""
        from metrics import predict_labels

        scores = self.predict_scores(x_raw, raw_names)
        rate = top_rate_override or self.top_rate
        threshold = np.quantile(scores, 1.0 - rate)
        return predict_labels(scores, threshold)

    def save(self, path):
        """Export complete pipeline state to a portable .npz archive."""
        d = {
            "version": self.version,
            "with_leakage": self.with_leakage,
            "top_rate": self.top_rate,
            "gbm_weight": self.gbm_weight,
            "risk_tables": self.risk_tables,
            "names": self.names,
            "gbm_models": [m.to_dict() for m in self.gbm_models],
            "mlp_state": self.mlp_state,
            "flags_stats": self.flags_stats,
            "mlp_model": self.mlp_model.to_dict(),
        }
        ModelSerializer.save(path, d)

    @classmethod
    def load(cls, path):
        """Instantiate a fully functional ensemble from an .npz archive."""
        d = ModelSerializer.load(path)
        inst = cls(
            with_leakage=d["with_leakage"],
            top_rate=d["top_rate"],
            gbm_weight=d["gbm_weight"],
            version=d.get("version", "1.0"),
        )
        inst.risk_tables = d["risk_tables"]
        inst.names = d["names"]
        inst.gbm_models = [GradientBoostedTrees.from_dict(m) for m in d["gbm_models"]]
        inst.mlp_state = d["mlp_state"]
        inst.flags_stats = d["flags_stats"]
        inst.mlp_model = MLPEnsemble.from_dict(d["mlp_model"])
        return inst


# ==============================================================================
# Backward Compatibility Namespaces
# ==============================================================================


class gbm:
    """Namespace providing backward compatibility with former gbm.py module."""

    DEFAULT_PARAMS = DEFAULT_GBM_PARAMS
    boost = staticmethod(boost)
    train = staticmethod(train_gbm)
    score = staticmethod(score_gbm)
    fit_bins = staticmethod(fit_bins)
    apply_bins = staticmethod(apply_bins)
    stratified_split = staticmethod(stratified_split)
    tree_features = staticmethod(tree_features)


class mlp:
    """Namespace providing backward compatibility with former mlp.py module."""

    DEFAULT_PARAMS = DEFAULT_MLP_PARAMS
    train = staticmethod(train_mlp)
    score = staticmethod(score_mlp)
    train_one = staticmethod(train_one_mlp)
