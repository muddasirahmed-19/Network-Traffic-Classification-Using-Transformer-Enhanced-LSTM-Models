"""
evaluate.py
-----------
Loads a trained model (deep learning or classical) plus its test split,
and produces everything the CEP requires for evaluation:

  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix (printed + saved as a PNG heatmap)
  - Accuracy curve and loss curve (deep learning models only — classical
    models like Logistic Regression / SVM don't train over epochs, so
    there's no curve to plot for them)
  - Appends a row to results/metrics.txt so every model's evaluation
    run accumulates into one comparison table over time.

Run:
    python src/evaluate.py --dataset darknet2020 --model transformer_lstm
    python src/evaluate.py --dataset darknet2020 --model svm
"""

import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras

from train import CLASSICAL_MODELS, DEEP_LEARNING_MODELS, load_processed_data


def get_predictions(model_name: str, args, data):
    X_test = data["X_test"]
    y_test = data["y_test"]

    if model_name in DEEP_LEARNING_MODELS:
        model_path = os.path.join(args.model_dir, f"{args.dataset}_{model_name}.keras")
        model = keras.models.load_model(model_path)
        y_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        model_path = os.path.join(args.model_dir, f"{args.dataset}_{model_name}.joblib")
        model = joblib.load(model_path)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        y_pred = model.predict(X_test_flat)
        # Not every classical model config exposes predict_proba; fall
        # back to the hard predictions for AUC if it's unavailable.
        y_prob = (
            model.predict_proba(X_test_flat)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred.astype(float)
        )

    return y_test, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=("NonTor", "Tor")):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Saved confusion matrix to {save_path}")


def plot_training_curves(history_path, model_name, dataset_name, figures_dir):
    """Only applicable to deep learning models, which have an epoch-by-epoch history."""
    if not os.path.exists(history_path):
        print(f"[plot] No training history found at {history_path}, skipping curves "
              f"(expected for classical ML models like SVM/Logistic Regression).")
        return

    with open(history_path) as f:
        history = json.load(f)

    os.makedirs(figures_dir, exist_ok=True)

    # Accuracy curve
    plt.figure(figsize=(6, 4))
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve — {model_name} on {dataset_name}")
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(figures_dir, f"{dataset_name}_{model_name}_accuracy_curve.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"[plot] Saved accuracy curve to {acc_path}")

    # Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve — {model_name} on {dataset_name}")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(figures_dir, f"{dataset_name}_{model_name}_loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"[plot] Saved loss curve to {loss_path}")


def append_metrics_to_file(metrics: dict, model_name: str, dataset_name: str, metrics_path: str):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    file_exists = os.path.exists(metrics_path)

    with open(metrics_path, "a") as f:
        if not file_exists:
            f.write(f"{'Dataset':<15}{'Model':<20}{'Accuracy':<10}{'Precision':<11}"
                    f"{'Recall':<9}{'F1':<9}{'ROC-AUC':<9}\n")
            f.write("-" * 85 + "\n")
        f.write(
            f"{dataset_name:<15}{model_name:<20}"
            f"{metrics['accuracy']:<10.4f}{metrics['precision']:<11.4f}"
            f"{metrics['recall']:<9.4f}{metrics['f1_score']:<9.4f}{metrics['roc_auc']:<9.4f}\n"
        )
    print(f"[metrics] Appended results to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--dataset", required=True, choices=["darknet2020", "iscxtor2016"])
    parser.add_argument(
        "--model", required=True,
        choices=sorted(DEEP_LEARNING_MODELS | CLASSICAL_MODELS),
    )
    parser.add_argument("--processed_dir", default="../dataset/processed")
    parser.add_argument("--model_dir", default="../results/models")
    parser.add_argument("--results_dir", default="../results")
    parser.add_argument("--figures_dir", default="../figures")
    args = parser.parse_args()

    data = load_processed_data(args.processed_dir, args.dataset)
    y_true, y_pred, y_prob = get_predictions(args.model, args, data)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"\n===== Results: {args.model} on {args.dataset} =====")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")

    cm_path = os.path.join(args.results_dir, f"{args.dataset}_{args.model}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)

    history_path = os.path.join(args.model_dir, f"{args.dataset}_{args.model}_history.json")
    plot_training_curves(history_path, args.model, args.dataset, args.figures_dir)

    metrics_txt_path = os.path.join(args.results_dir, "metrics.txt")
    append_metrics_to_file(metrics, args.model, args.dataset, metrics_txt_path)


if __name__ == "__main__":
    main()
