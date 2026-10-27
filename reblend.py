"""Reblend saved test components into a submission without fitting models."""
import argparse
from pathlib import Path
import numpy as np
from helpers import create_csv_submission
from metrics import predict_labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--components', required=True)
    parser.add_argument('--gbm_weight', type=float, default=0.55)
    parser.add_argument('--top_rate', type=float, default=0.129)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.gbm_weight <= 1 or not 0 < args.top_rate < 1:
        parser.error('Weight must be in [0, 1] and rate in (0, 1)')
    with np.load(args.components, allow_pickle=False) as data:
        if not {'ids', 'gbm', 'mlp', 'boost'} <= set(data.files):
            parser.error('Expected test components from run.py, including test IDs')
        ids, g, m, boost = (data[k] for k in ('ids', 'gbm', 'mlp', 'boost'))
        if not (ids.ndim == 1 and len(ids) > 0 and g.shape == m.shape == boost.shape == ids.shape):
            parser.error('Components must be aligned, nonempty 1D vectors')
        if not all(np.isfinite(a).all() for a in (ids, g, m, boost)) or len(np.unique(ids)) != len(ids):
            parser.error('Components contain nonfinite scores or duplicate IDs')
        scores = args.gbm_weight * g + (1 - args.gbm_weight) * m + boost
        labels = predict_labels(scores, np.quantile(scores, 1 - args.top_rate))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        create_csv_submission(ids, labels, args.output)
        print(f'Wrote {args.output}: {len(labels)} rows, {np.mean(labels == 1):.4%} positive')


if __name__ == '__main__':
    main()
