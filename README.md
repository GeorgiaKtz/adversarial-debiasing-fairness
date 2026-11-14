# Adversarial Debiasing on Adult, German and COMPAS

This repository applies **Adversarial Debiasing** (Zhang, Lemoine & Mitchell, 2018, *Mitigating Unwanted Biases with Adversarial Learning*) to three standard fairness datasets. It compares fairness before and after mitigation.

Adversarial debiasing is an **in-processing** method. A predictor network is trained together with an adversary that tries to predict the protected attribute from the predictor's output, and the predictor learns to be accurate while making the adversary fail. The implementation used is `AdversarialDebiasing` from [IBM AI Fairness 360](https://github.com/Trusted-AI/AIF360). In AIF360 the adversary sees both the prediction and the true label, so it targets **equality of odds**.

## Datasets

| Dataset | Task (favourable label = 1) | Protected attribute | Privileged group |
|---|---|---|---|
| Adult (45,222 rows) | income > 50K | sex | Male |
| German credit (1,000 rows) | good credit risk | age | older than 25 |
| COMPAS (6,172 rows) | no re-offence within 2 years | race | Caucasian |

The notebook downloads the raw files automatically into a `data/` folder.

## Pipeline

1. Load the datasets and define the label and protected attribute.
2. Compute **DP_Diff**, **DIR** and **consistency** on the original labels, before any training.
3. Split into train and test sets (70/30), stratified on label × protected group.
4. Standardise the features, with the scaler fitted on the training set only.
5. Train a baseline network without the adversary and the adversarially debiased network.
6. Compute fairness and performance metrics on the held-out test set.
7. Compare the metrics before and after mitigation, as a table and a chart.

## Parameter settings

| Parameter | Value |
|---|---|
| Hidden units (predictor) | 200 |
| Adversary loss weight (α) | 0.1 |
| Epochs | 50 |
| Batch size | 128 |
| Optimiser | Adam, learning rate 0.001 with exponential decay |
| Test size | 0.3 (stratified) |
| Consistency neighbours (k) | 5 |
| Random seed | 42 |

## Results (test set)

| Dataset | Labels / model | DP_Diff | DIR | Consistency | Accuracy |
|---|---|---|---|---|---|
| Adult | Original labels | −0.1989 | 0.3636 | 0.8240 | |
| | Baseline NN | −0.1898 | 0.3326 | 0.9375 | 0.8468 |
| | **Adversarial debiasing** | **−0.1013** | **0.5675** | 0.9416 | 0.8437 |
| German | Original labels | −0.1494 | 0.7948 | 0.7533 | |
| | Baseline NN | −0.0615 | 0.9230 | 0.8287 | 0.8000 |
| | **Adversarial debiasing** | **0.1466** | **1.2108** | 0.8027 | 0.7900 |
| COMPAS | Original labels | −0.0967 | 0.8411 | 0.6614 | |
| | Baseline NN | −0.1979 | 0.7326 | 0.9578 | 0.6836 |
| | **Adversarial debiasing** | **−0.0645** | **0.8989** | 0.9610 | 0.6712 |

Ideal values: DP_Diff = 0, DIR = 1, consistency = 1.

![Fairness comparison](fairness_comparison.png)

**Summary of the results:**

- On Adult and COMPAS, debiasing moved DP_Diff and DIR clearly towards their ideal values, for at most 1.2 points of accuracy.
- On the small German dataset the method over-corrected: DP_Diff changed sign and DIR rose above 1.
- Consistency rose mostly because model predictions are smoother than the original labels. The baseline network shows almost the same rise.

Exact numbers can vary slightly with the TensorFlow version.

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then open `adversarial_debiasing.ipynb` in VS Code or Jupyter, select the `.venv` kernel and run all cells. Training takes about 1–2 minutes on a CPU.

## Files

- `adversarial_debiasing.ipynb`: the full mitigation pipeline
- `requirements.txt`: Python packages needed
- `fairness_comparison.png`: chart of the fairness measures before and after mitigation
- `results_table.png`: results table as an image

## Reference

B. H. Zhang, B. Lemoine and M. Mitchell. *Mitigating Unwanted Biases with Adversarial Learning.* AAAI/ACM Conference on AI, Ethics, and Society (AIES), 2018. [arXiv:1801.07593](https://arxiv.org/abs/1801.07593)
