# Network Traffic Classification Using Transformer-Enhanced LSTM Models

Computer Networks Semester Project — Transformer-LSTM hybrid model for
classifying Tor (Darknet) vs. Non-Tor network traffic, built to improve on
a CNN-LSTM baseline from the literature.

## 1. Project Description

This project classifies encrypted network traffic as **Tor (Darknet)** or
**Non-Tor (benign)** using a deep learning model that combines an LSTM
layer (for learning sequential/temporal patterns in traffic flows) with a
Transformer self-attention layer (for learning which flow features matter
most, and capturing long-range relationships between them).

The base paper we are improving on is:

> Mandela, N., Sonia, Mistry, N., Nagpal, A. (2025). *Efficient Dark Web
> traffic classification using a hybrid CNN-LSTM model.* International
> Journal of Information Technology.

That paper used a **CNN + LSTM** hybrid on the CIC-Darknet2020 dataset and
reported 98.39% accuracy. Our project replaces the CNN feature-extraction
stage with a **Transformer multi-head self-attention** stage, while
keeping the LSTM for temporal modeling, and benchmarks the result against
that paper's own baselines (Logistic Regression, SVM, CNN, LSTM) plus a
CNN-LSTM model we reproduce ourselves.

## 2. Datasets

| # | Dataset | Source | Role |
|---|---|---|---|
| 1 | **CIC-Darknet2020** | [Kaggle](https://www.kaggle.com/datasets/dhoogla/cicdarknet2020) | Primary dataset — same dataset as the base paper |
| 2 | **ISCXTor2016 (Tor-nonTor)** | [UNB CIC](https://www.unb.ca/cic/datasets/tor.html) | Second dataset for cross-dataset comparison |

Both datasets were produced by the same lab (Canadian Institute for
Cybersecurity) using the same flow-feature extractor (CICFlowMeter /
ISCXFlowMeter), so they share a near-identical set of ~80 flow-level
statistical features (flow duration, packet length stats, IAT stats,
byte/sec rates, etc). This makes a clean, fair, apples-to-apples
comparison possible without redesigning the pipeline for each dataset.

See `dataset/dataset_link.txt` for exact download links.

## 3. Installation

```bash
git clone <your-repo-url>
cd network-traffic-transformer-lstm
pip install -r requirements.txt
```

Designed to run on **Google Colab free tier** (or any machine with
~4GB+ RAM and optionally a GPU). No paid compute required.

## 4. How to Run

1. Download both datasets (see `dataset/dataset_link.txt`) and place the
   CSV files in `dataset/raw/` as `darknet2020.csv` and `iscxtor2016.csv`.
2. From inside the `src/` folder, run preprocessing:
   ```bash
   cd src
   python preprocessing.py --dataset darknet2020
   python preprocessing.py --dataset iscxtor2016
   ```
3. Train and evaluate each model, from inside `src/`. Repeat for every
   model (`logistic_regression`, `svm`, `lstm_only`, `cnn_only`,
   `cnn_lstm`, `transformer_lstm`) and for both datasets:
   ```bash
   python train.py --dataset darknet2020 --model transformer_lstm
   python evaluate.py --dataset darknet2020 --model transformer_lstm
   ```
   For the proposed model (`transformer_lstm`) on the primary dataset
   (`darknet2020`), add `--primary` to the evaluate command so the
   confusion matrix and accuracy/loss curves are also saved under the
   exact filenames this repo structure specifies
   (`results/confusion_matrix.png`, `figures/accuracy_curve.png`,
   `figures/loss_curve.png`):
   ```bash
   python evaluate.py --dataset darknet2020 --model transformer_lstm --primary
   ```
   Each `evaluate.py` run also appends a row to `results/metrics.txt`,
   so running this for every model builds up the full comparison table.

Or open `notebooks/experiment.ipynb` to run the entire pipeline
end-to-end interactively (recommended for Google Colab).

## 5. Model Architecture

```
Input (flow features, reshaped to sequence)
        ↓
LSTM layer (sequential / temporal patterns)
        ↓
Transformer encoder block (multi-head self-attention)
        ↓
Dropout
        ↓
Dense (ReLU)
        ↓
Output Dense (1 unit, Sigmoid) — Tor vs Non-Tor
```

Full implementation: `src/model.py`.

## 6. Baselines Compared

- Logistic Regression
- Support Vector Machine (SVM)
- Standalone LSTM
- Standalone CNN
- CNN + LSTM (reproduction of the base paper's architecture)
- **Transformer + LSTM (proposed model)**

## 7. Evaluation Metrics

Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC,
training/validation accuracy curves, training/validation loss curves.
Results are saved to `results/metrics.txt` and `results/confusion_matrix.png`;
curves are saved to `figures/`.

## 8. Results Summary

_Fill in after running training — see `results/metrics.txt` for the
actual numbers produced by your run. Do not copy numbers from the base
paper into this section; only your own experimental results belong here._

## 9. Repository Structure

```
network-traffic-transformer-lstm/
├── README.md
├── requirements.txt
├── dataset/
│   └── dataset_link.txt
├── notebooks/
│   └── experiment.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── results/
│   ├── metrics.txt
│   └── confusion_matrix.png
├── figures/
│   ├── accuracy_curve.png
│   └── loss_curve.png
└── report/
    └── final_report.pdf
```

## 10. Team Members

- _Name 1_
- _Name 2_
- _Name 3_

## 11. Academic Integrity Note

All results in this repository come from our own implementation and our
own training runs. AI assistance was used for coding, debugging, and
understanding concepts (as permitted by the course rules); the report
text and analysis are written by the team in our own words.
