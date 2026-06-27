# Network Traffic Classification Using Transformer-Enhanced LSTM Models

This is our Computer Networks semester project. We built a deep learning
model that can look at network traffic and decide whether it is **Tor
(Darknet) traffic** or **normal (Non-Tor) traffic**.

## 1. What This Project Does

There is an existing research paper that solved this same problem using
a CNN + LSTM model and got about 98.39% accuracy:

> Mandela, N., Sonia, Mistry, N., Nagpal, A. (2025). Efficient Dark Web
> traffic classification using a hybrid CNN-LSTM model. International
> Journal of Information Technology.

Our teacher asked us to not copy this model, but to try to improve the
idea. So instead of using a CNN, we used a **Transformer attention
layer** together with the LSTM. The idea is that a CNN can only look at
features that are next to each other, but attention can connect any two
features to each other, which might help the model find patterns the
CNN would miss.

We compared our model against 5 simpler models to check if it actually
helps: Logistic Regression, SVM, a plain LSTM, a plain CNN, and a
CNN-LSTM model (we built this one ourselves to copy the base paper's
design, just so we have something fair to compare against).

## 2. Datasets We Used

| # | Dataset | Where we got it |
|---|---|---|
| 1 | CIC-Darknet2020 | [Kaggle](https://www.kaggle.com/datasets/dhoogla/cicdarknet2020) |
| 2 | ISCXTor2016 (Scenario-A, Tor vs Non-Tor) | [UNB CIC website](https://www.unb.ca/cic/datasets/tor.html) |

We used the first dataset because it's the same one the base paper
used. We used the second one because our teacher told us to compare on
at least 2 datasets, and this one was made by the same research lab, so
it has similar columns and was easier to use with the same code.

Download links are also in `dataset/dataset_link.txt`.

## 3. How to Set It Up

```bash
git clone <our-repo-url>
cd network-traffic-transformer-lstm
pip install -r requirements.txt
```

We ran everything on Google Colab (free version), since none of us have
a strong GPU on our own laptops.

## 4. How to Run This Project

1. Download both datasets (links above) and put them inside
   `dataset/raw/` as `darknet2020.csv` (or `.parquet`) and
   `iscxtor2016.csv`.
2. Go into the `src` folder and clean the data:
   ```bash
   cd src
   python preprocessing.py --dataset darknet2020
   python preprocessing.py --dataset iscxtor2016
   ```
3. Train and test each model. You repeat this for every model name
   (`logistic_regression`, `svm`, `lstm_only`, `cnn_only`, `cnn_lstm`,
   `transformer_lstm`) and for both datasets:
   ```bash
   python train.py --dataset darknet2020 --model transformer_lstm
   python evaluate.py --dataset darknet2020 --model transformer_lstm
   ```
   For our main model (`transformer_lstm`) on our main dataset
   (`darknet2020`), we added `--primary` so the confusion matrix and
   accuracy/loss graphs get saved with the same names our teacher's
   project structure asked for:
   ```bash
   python evaluate.py --dataset darknet2020 --model transformer_lstm --primary
   ```
   Every time we run `evaluate.py`, it adds one line to
   `results/metrics.txt`, so by the end this file has the results for
   every model we tested.

We also made a Colab notebook (`notebooks/experiment.ipynb`) that does
all of this step by step, which is what we actually used.

## 5. Our Model's Structure

```
Input (traffic flow features)
        ↓
LSTM layer
        ↓
Transformer attention layer
        ↓
Dropout (to avoid overfitting)
        ↓
Dense layer
        ↓
Output (Tor or Non-Tor)
```

The code for this is in `src/model.py`.

## 6. Models We Compared

- Logistic Regression
- SVM
- LSTM only
- CNN only
- CNN + LSTM (this is our version of the base paper's model)
- Transformer + LSTM (our model)

## 7. How We Measured Performance

We used Accuracy, Precision, Recall, F1-score, Confusion Matrix, and
ROC-AUC, plus accuracy/loss graphs over training epochs. These get
saved automatically when we run `evaluate.py` — results go into
`results/`, graphs go into `figures/`.

## 8. Our Results

See `results/metrics.txt` for the real numbers from our own training
runs. We did not copy any numbers from the base paper here — everything
in that file came from actually running our code.

## 9. Project Folder Structure

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

## 10. Group Members

- Muddasir Ahmed
- Bilal Sultan
- Ayesha Khan
- Abdal Ahmed

## 11. Note on AI Usage

We used AI tools to help us with coding and debugging, and to help us
understand things we were stuck on, which our course rules say is
allowed. The report itself and all written explanations are in our own
words. All results in this repo are from our own training runs, not
copied or made up.
