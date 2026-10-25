"""Gradient boosted decision trees for binary classification (numpy only).

Link with the course:
    * Loss: the same logistic loss as logistic regression,
      L(F) = sum_n log(1 + exp(F(x_n))) - y_n F(x_n), but the model F is a
      sum of small regression trees instead of a linear function x^T w.
    * Boosting is gradient descent in function space: each round adds one
      tree that approximates a step against the gradient of L, scaled by a
      learning rate (like the step size gamma of GD).
    * Newton step: as in Newton's method for logistic regression we use the
      gradient g = p - y and the hessian h = p (1 - p) of the loss. The best
      constant in a leaf is the Newton step -G / (H + l2), and a split is
      scored by how much it lowers the second-order approximation of L.
    * Regularization (bias-variance): shallow trees, a minimum number of
      rows per leaf, an L2 penalty on the leaf values (like ridge),
      shrinkage, row / column subsampling and early stopping on a
      validation split all reduce the variance.

Features are cut into quantile bins (fitted on the training rows); NaN gets
its own bin and every split learns on which side the NaN rows go. The sums
of g and h per (feature, bin) are computed with np.bincount.
"""

import numpy as np

DEFAULT_PARAMS = {
    "n_trees": 1500,  # maximum number of boosting rounds
    "learning_rate": 0.05,  # shrinkage of every tree
    "max_depth": 5,  # depth of each tree (at most 2**max_depth leaves)
    "min_leaf": 200,  # min number of rows in a leaf
    "min_hessian": 1.0,  # min sum of h in a leaf
    "l2": 5.0,  # L2 penalty on the leaf values
    "min_gain": 0.0,  # min loss decrease to split a node
    "n_bins": 64,  # quantile bins per feature (+1 bin for NaN)
    "subsample": 0.7,  # share of the rows used by each tree
    "colsample": 0.8,  # share of the features used by each tree
    "valid_frac": 0.2,  # inner validation split for early stopping
    "patience": 100,  # stop after this many rounds without improvement
    "target_enc_last": 100,  # smoothing m of the target encoding of the last
    # column (the state code _STATE)
    "seed": 42,
}

# Answers given only by people with a heart history (they leak the label).
# clean_data already drops them; the guard makes sure they never come back.
LEAKY = [
    "HAREHAB1",
    "CVDASPRN",
    "ASPUNSAF",
    "RLIVPAIN",
    "RDUCHART",
    "RDUCSTRK",
    "STREHAB1",
]


def sigmoid(t):
    """Logistic function, numerically stable."""
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def logistic_loss(y01, f):
    """Mean logistic loss of the logits f for labels y01 in {0, 1}."""
    return float(np.mean(np.logaddexp(0.0, f) - y01 * f))


def tree_features(xc, names):
    """Input of the trees: the cleaned matrix without the leaky columns.

    Trees need no imputation, one-hot encoding or standardization (NaN has
    its own bin). The raw state code _STATE must be the last column
    (preprocessing.clean_data with keep_raw=("_STATE",)).

    Args:
        xc: cleaned matrix (N, D), NaN allowed.
        names: list of the D column names.

    Returns:
        numpy array (N, D') with _STATE still last.
    """
    assert names[-1] == "_STATE"
    keep = np.array([n not in LEAKY for n in names])
    return np.asarray(xc)[:, keep]


# ---------------------------------------------------------------- inputs
def stratified_split(y, frac, seed):
    """Indices (train, valid) with a share `frac` of each class in valid."""
    rng = np.random.default_rng(seed)
    va = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        va.append(idx[: int(round(frac * len(idx)))])
    va = np.sort(np.concatenate(va))
    tr = np.setdiff1d(np.arange(len(y)), va)
    return tr, va


def fit_target_encoding(codes, y01, m):
    """Smoothed prevalence of each category (here: each state).

    value(c) = (positives in c + m * prior) / (rows in c + m): small groups
    are shrunk towards the global prior (a simple regularization).

    Returns:
        (sorted codes, values, prior).
    """
    prior = float(np.mean(y01))
    cats, inv = np.unique(codes, return_inverse=True)
    pos = np.bincount(inv, weights=y01)
    cnt = np.bincount(inv)
    return cats, (pos + m * prior) / (cnt + m), prior


def apply_target_encoding(tx, encoding):
    """Replace the last column of tx (category codes) by their encoding.

    Unknown codes get the prior.
    """
    cats, values, prior = encoding
    codes = tx[:, -1]
    pos = np.clip(np.searchsorted(cats, codes), 0, len(cats) - 1)
    enc = np.where(cats[pos] == codes, values[pos], prior)
    return np.column_stack([tx[:, :-1], enc])


def fit_bins(x, n_bins):
    """Bin edges per feature (quantiles of the non-NaN values).

    Args:
        x: numpy array (N, D), may contain NaN.
        n_bins: maximum number of bins for the non-NaN values.

    Returns:
        list of D sorted arrays of edges (at most n_bins - 1 each).
    """
    edges = []
    for j in range(x.shape[1]):
        col = x[:, j]
        values = np.unique(col[~np.isnan(col)])
        if len(values) <= n_bins:
            # few distinct values: one bin per value (cut in the middle)
            edges.append((values[:-1] + values[1:]) / 2)
        else:
            qs = np.linspace(0, 1, n_bins + 1)[1:-1]
            edges.append(np.unique(np.quantile(col[~np.isnan(col)], qs)))
    return edges


def apply_bins(x, edges, n_bins):
    """Bin index of every entry: 0..n_bins-1 for values, n_bins for NaN.

    Returns:
        uint8 numpy array (N, D).
    """
    xb = np.empty(x.shape, dtype=np.uint8)
    for j, e in enumerate(edges):
        col = x[:, j]
        xb[:, j] = np.searchsorted(e, col, side="right")
        xb[np.isnan(col), j] = n_bins
    return xb


# ---------------------------------------------------------------- one tree
def build_histograms(xb, g, h, rows, slot, n_slots, n_bins):
    """Sums of g, h and counts per (node slot, feature, bin), via bincount.

    Args:
        xb: binned features (N, D).
        g, h: gradients and hessians (N,).
        rows: indices of the rows to use.
        slot: node slot (0..n_slots-1) of each row in `rows`.
        n_slots: number of nodes.
        n_bins: number of value bins (the NaN bin is n_bins).

    Returns:
        G, H, C: arrays (n_slots, D, n_bins + 1).
    """
    d, nb = xb.shape[1], n_bins + 1
    size = n_slots * nb
    G, H, C = [np.empty((d, n_slots, nb)) for _ in range(3)]
    g_r, h_r = g[rows], h[rows]
    sub = np.ascontiguousarray(xb[rows].T)  # (D, n) binned values of the rows
    base = slot * nb
    for j in range(d):
        # flat index of (node slot, bin) for feature j, one bincount per sum
        idx = base + sub[j]
        G[j] = np.bincount(idx, g_r, size).reshape(n_slots, nb)
        H[j] = np.bincount(idx, h_r, size).reshape(n_slots, nb)
        C[j] = np.bincount(idx, minlength=size).reshape(n_slots, nb)
    # reorder as (node slot, feature, bin)
    G, H, C = [A.transpose(1, 0, 2) for A in (G, H, C)]
    return G, H, C


def best_splits(G, H, C, col_mask, p):
    """Best split of every node from its histograms.

    A split sends bins <= b to the left and bins > b to the right; the NaN
    bin goes left or right (both are tried). The gain is the decrease of
    the second-order approximation of the loss:
        GL^2 / (HL + l2) + GR^2 / (HR + l2) - G^2 / (H + l2).

    Returns:
        feat, thr, nan_left, gain, n_left: arrays (n_slots,).
    """
    l2, nv = p["l2"], G.shape[2] - 1
    # cumulative sums over the value bins, then add the NaN bin or not
    stats = []
    for A in (G, H, C):
        cum = np.cumsum(A[:, :, :nv], axis=2)
        nan = A[:, :, nv:]
        stats.append(np.stack([cum, cum + nan], axis=3))  # (S, D, nv, 2)
    GL, HL, CL = stats
    Gt, Ht, Ct = [A[:, 0, :].sum(axis=1)[:, None, None, None] for A in (G, H, C)]
    GR, HR, CR = Gt - GL, Ht - HL, Ct - CL
    gain = GL**2 / (HL + l2) + GR**2 / (HR + l2) - Gt**2 / (Ht + l2)
    # regularization: enough rows and curvature on both sides
    ok = (CL >= p["min_leaf"]) & (CR >= p["min_leaf"])
    ok &= (HL >= p["min_hessian"]) & (HR >= p["min_hessian"])
    ok &= col_mask[None, :, None, None]
    gain = np.where(ok, gain, -np.inf)
    n_slots = G.shape[0]
    best = np.argmax(gain.reshape(n_slots, -1), axis=1)
    feat, thr, nan_left = np.unravel_index(best, gain.shape[1:])
    pick = (np.arange(n_slots), feat, thr, nan_left)
    return feat, thr, nan_left, gain[pick], CL[pick]


def child_histograms(xb, g, h, rows, slot, hists, par, child_slot, n_left, n_bins):
    """Histograms of the next level with the "subtraction trick".

    Only the smaller child of every split node is scanned; the larger child
    is parent - smaller child, so about half of the rows are read.

    Args:
        rows, slot: rows of the next level and their child slot.
        hists: (G, H, C) of the current level.
        par: slots of the split nodes; child_slot: (left, right) slots.
        n_left: rows going left in every node of the current level.

    Returns:
        G, H, C of the next level.
    """
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
    """Grow one tree level by level on the given rows.

    Returns:
        tree: dict of arrays feat, thr, nan_left, left, right, value
            (leaves have feat = -1).
    """
    nbin, lr = p["n_bins"], p["learning_rate"]
    keys = ("feat", "thr", "nan_left", "left", "right", "value")
    tree = {k: [] for k in keys}

    def new_node():
        for k, v in zip(keys, (-1, 0, 0, 0, 0, 0.0)):
            tree[k].append(v)
        return len(tree["feat"]) - 1

    level = [new_node()]  # node ids of the current level
    slot = np.zeros(len(rows), dtype=np.int64)  # level slot of each row
    G, H, C = build_histograms(xb, g, h, rows, slot, 1, nbin)
    for depth in range(p["max_depth"] + 1):
        Gt, Ht = G[:, 0, :].sum(axis=1), H[:, 0, :].sum(axis=1)
        split = np.zeros(len(level), dtype=bool)
        if depth < p["max_depth"]:
            f, b, nl, gain, n_left = best_splits(G, H, C, col_mask, p)
            split = gain > p["min_gain"]
        # leaves: Newton step -G / (H + l2), shrunk by the learning rate
        for s in np.where(~split)[0]:
            tree["value"][level[s]] = -lr * Gt[s] / (Ht[s] + p["l2"])
        if not split.any():
            break
        # create the children of the split nodes
        par = np.where(split)[0]
        next_level, child_slot = [], -np.ones((len(level), 2), dtype=np.int64)
        for s in par:
            node = level[s]
            tree["feat"][node], tree["thr"][node] = int(f[s]), int(b[s])
            tree["nan_left"][node] = int(nl[s])
            tree["left"][node], tree["right"][node] = new_node(), new_node()
            child_slot[s] = [len(next_level), len(next_level) + 1]
            next_level += [tree["left"][node], tree["right"][node]]
        # send each row of a split node to its left or right child
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
    """Value of the leaf reached by every row of xb."""
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


# ---------------------------------------------------------------- boosting
def boost(xb, y01, xb_val, y_val, p):
    """Gradient boosting with Newton leaves and early stopping.

    Args:
        xb, y01: binned training features and labels in {0, 1}.
        xb_val, y_val: validation data for early stopping.
        p: parameters (see DEFAULT_PARAMS).

    Returns:
        f0: initial logit (log-odds of the prior), trees: list of trees cut
        at the round with the lowest validation loss.
    """
    rng = np.random.default_rng(p["seed"])
    n, d = xb.shape
    prior = np.clip(y01.mean(), 1e-6, 1 - 1e-6)
    f0 = float(np.log(prior / (1 - prior)))
    f, f_val = np.full(n, f0), np.full(len(xb_val), f0)
    trees, val_curve = [], []
    for t in range(p["n_trees"]):
        # gradient and hessian of the logistic loss at the current logits
        prob = sigmoid(f)
        g, h = prob - y01, np.maximum(prob * (1 - prob), 1e-12)
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
        # early stopping on the validation loss
        f_val += predict_tree(tree, xb_val, p["n_bins"])
        val_curve.append(logistic_loss(y_val, f_val))
        if t - int(np.argmin(val_curve)) >= p["patience"]:
            break
    return f0, trees[: int(np.argmin(val_curve)) + 1]


def train(y, tx, params=None):
    """Train the boosted trees.

    The rows are split into an inner train / validation part (stratified,
    `valid_frac`). The state prevalence (last column) is fitted on the inner
    train labels only. Boosting stops when the validation loss has not
    improved for `patience` rounds and the model is cut at the best round.

    Args:
        y: labels in {-1, 1}, shape (N,).
        tx: tree_features output (N, D), NaN allowed, _STATE last.
        params: dict, missing keys come from DEFAULT_PARAMS.

    Returns:
        model: dict (target encoding, bin edges, f0, trees).
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    y01 = (y == 1).astype(np.float64)
    tr, va = stratified_split(y01, p["valid_frac"], p["seed"])
    encoding = fit_target_encoding(tx[tr, -1], y01[tr], p["target_enc_last"])
    tx = apply_target_encoding(tx, encoding)
    edges = fit_bins(tx, p["n_bins"])
    xb = apply_bins(tx, edges, p["n_bins"])
    f0, trees = boost(xb[tr], y01[tr], xb[va], y01[va], p)
    model = {"encoding": encoding, "edges": edges, "n_bins": p["n_bins"]}
    model.update(f0=f0, trees=trees)
    return model


def score(model, tx):
    """Logit of every row (higher = more likely positive)."""
    tx = apply_target_encoding(tx, model["encoding"])
    xb = apply_bins(tx, model["edges"], model["n_bins"])
    f = np.full(len(tx), model["f0"])
    for tree in model["trees"]:
        f += predict_tree(tree, xb, model["n_bins"])
    return f
