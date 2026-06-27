"""
train.py
--------
Trains a chosen model on a chosen preprocessed dataset, and saves:
  - the trained model (Keras .keras file, or sklearn .joblib file)
  - training history (accuracy/loss per epoch) for plotting later

Supports both:
  - Deep learning models from model.py (transformer_lstm, lstm_only,
    cnn_only, cnn_lstm) — trained with Keras.
  - Classical ML baselines (logistic_regression, svm) — trained with
    scikit-learn, using the same preprocessed train/test split so every
    model in the comparison table sees identical data.

Run:
    python src/train.py --dataset darknet2020 --model transformer_lstm
    python src/train.py --dataset darknet2020 --model svm
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow import keras

from model import build_model

DEEP_LEARNING_MODELS = {"transformer_lstm", "lstm_only", "cnn_only", "cnn_lstm"}
CLASSICAL_MODELS = {"logistic_regression", "svm"}


def load_processed_data(processed_dir: str, dataset_name: str):
    path = os.path.join(processed_dir, f"{dataset_name}_processed.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Run preprocessing.py for this dataset first."
        )
    data = np.load(path, allow_pickle=True)
    return data


def train_deep_learning_model(model_name: str, data, args):
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    input_shape = X_train.shape[1:]  # (timesteps, num_features)
    model = build_model(model_name, input_shape)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(model.summary())

    # EarlyStopping prevents wasting Colab GPU time once validation loss
    # stops improving, and restores the best-performing weights rather
    # than whatever the last epoch happened to land on.
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - start
    print(f"[train] Training took {elapsed:.1f} seconds")

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{args.dataset}_{model_name}.keras")
    model.save(model_path)
    print(f"[train] Saved model to {model_path}")

    history_path = os.path.join(args.model_dir, f"{args.dataset}_{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    print(f"[train] Saved training history to {history_path}")


def train_classical_model(model_name: str, data, args):
    # Classical sklearn models expect 2D input (samples, features), so
    # we flatten away the timesteps dimension (which is 1 anyway).
    X_train = data["X_train"].reshape(data["X_train"].shape[0], -1)
    y_train = data["y_train"]

    if model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "svm":
        # linear kernel keeps training time reasonable on Colab's free
        # CPU/GPU for datasets with tens of thousands of rows; an RBF
        # kernel SVM would be far slower at this scale.
        clf = SVC(kernel="linear", probability=True, random_state=42)
    else:
        raise ValueError(f"Unknown classical model '{model_name}'")

    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"[train] Training took {elapsed:.1f} seconds")

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{args.dataset}_{model_name}.joblib")
    joblib.dump(clf, model_path)
    print(f"[train] Saved model to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a model on a preprocessed dataset.")
    parser.add_argument("--dataset", required=True, choices=["darknet2020", "iscxtor2016"])
    parser.add_argument(
        "--model", required=True,
        choices=sorted(DEEP_LEARNING_MODELS | CLASSICAL_MODELS),
    )
    parser.add_argument("--processed_dir", default="../dataset/processed")
    parser.add_argument("--model_dir", default="../results/models")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    data = load_processed_data(args.processed_dir, args.dataset)

    if args.model in DEEP_LEARNING_MODELS:
        train_deep_learning_model(args.model, data, args)
    else:
        train_classical_model(args.model, data, args)


if __name__ == "__main__":
    main()
