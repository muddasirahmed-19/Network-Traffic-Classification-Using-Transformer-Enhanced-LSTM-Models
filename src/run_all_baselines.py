"""
run_all_baselines.py
---------------------
Convenience script: trains and evaluates EVERY model (all baselines +
the proposed Transformer+LSTM model) on a single dataset in one run, so
you get a complete comparison table without typing ~12 separate
train/evaluate commands.

This directly produces the "Compare against baselines" requirement from
the CEP: Logistic Regression, SVM, LSTM, CNN, CNN-LSTM, and our proposed
Transformer-LSTM model, all evaluated identically and appended to
results/metrics.txt for an easy side-by-side comparison.

Run:
    python src/run_all_baselines.py --dataset darknet2020
    python src/run_all_baselines.py --dataset iscxtor2016

Note: this just calls train.py and evaluate.py under the hood for each
model in turn -- it doesn't duplicate their logic. If you want to retrain
a single model with different hyperparameters, use train.py/evaluate.py
directly instead.
"""

import argparse
import subprocess
import sys

ALL_MODELS = [
    "logistic_regression",
    "svm",
    "lstm_only",
    "cnn_only",
    "cnn_lstm",
    "transformer_lstm",  # proposed model, run last so it's freshest in metrics.txt
]


def run(cmd: list):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[run_all_baselines] Command failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate every model on one dataset.")
    parser.add_argument("--dataset", required=True, choices=["darknet2020", "iscxtor2016"])
    parser.add_argument("--epochs", type=int, default=40, help="Epochs for deep learning models only.")
    args = parser.parse_args()

    for model_name in ALL_MODELS:
        train_cmd = [sys.executable, "train.py", "--dataset", args.dataset, "--model", model_name]
        if model_name in {"lstm_only", "cnn_only", "cnn_lstm", "transformer_lstm"}:
            train_cmd += ["--epochs", str(args.epochs)]
        run(train_cmd)

        eval_cmd = [sys.executable, "evaluate.py", "--dataset", args.dataset, "--model", model_name]
        run(eval_cmd)

    print(f"\nAll models trained and evaluated for '{args.dataset}'. "
          f"See results/metrics.txt for the full comparison table.")


if __name__ == "__main__":
    main()
