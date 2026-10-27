import os
import argparse
import numpy as np
from metrics import predict_labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate submission CSV from precomputed test scores."
    )
    parser.add_argument(
        "--scores_path", default="submissions/test_scores_with_leakage.npy"
    )
    parser.add_argument("--data_path", default="data/dataset")
    parser.add_argument("--rate", type=float, default=0.1265)
    parser.add_argument("--output", default="submissions/submission_with_leakage.csv")
    args = parser.parse_args()

    scores = np.load(args.scores_path)
    # Load test ids
    test_ids_file = os.path.join(args.data_path, "test_ids_raw.npy")
    if os.path.exists(test_ids_file):
        test_ids = np.load(test_ids_file)
    else:
        with open(os.path.join(args.data_path, "x_test.csv")) as f:
            lines = [l.strip().split(",")[0] for l in list(f)[1:]]
        test_ids = np.array([int(x) for x in lines])

    cut = np.quantile(scores, 1.0 - args.rate)
    y_pred = predict_labels(scores, cut)
    pos_count = np.sum(y_pred == 1)

    with open(args.output, "w") as f:
        f.write("Id,Prediction\n")
        for tid, pred in zip(test_ids, y_pred):
            f.write(f"{int(tid)},{int(pred)}\n")

    print(
        f"Generated {args.output}: {len(y_pred)} rows, {pos_count} positives ({pos_count/len(y_pred)*100:.2f}%, rate={args.rate:.4f})"
    )


if __name__ == "__main__":
    main()
