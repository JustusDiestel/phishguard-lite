# PhishGuard-Lite

**PhishGuard-Lite** is a lightweight student project for phishing URL detection. It combines handcrafted URL features, feature selection, classical machine learning, an optional pretrained Hugging Face URL classifier, and a simple hybrid risk score.

Project title:

> PhishGuard-Lite: A Lightweight Hybrid Phishing Detection System Using Feature Selection and Pretrained Hugging Face Models

Academic topic:

> Lightweight Machine Learning-Based Phishing Detection with Feature Selection and Pretrained Models: A Practical Evaluation Study

This project is defensive and educational. It does not scrape phishing sites, collect credentials, or include real credential-stealing URLs.

## Features

- Single URL analysis
- CSV batch evaluation with `url,label` columns
- Handcrafted URL feature extraction
- Random Forest, Logistic Regression, Linear SVM, and Gradient Boosting training
- Feature selection with mutual information and permutation importance
- Optional Hugging Face pretrained URL model layer
- Real-dataset benchmark support for UCI PhiUSIIL
- Character TF-IDF URL text model
- Optional lightweight character-CNN deep learning model
- Hybrid score with transparent fallback behavior
- Streamlit interface for classroom presentation
- Report assets: confusion matrix, feature importance, feature selection chart, and CSV tables

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Train Models

```bash
python train.py
```

This trains the classical models on `data/sample_urls.csv` and saves model bundles in `models/`.

## Run the Streamlit App

```bash
streamlit run app.py
```

If no trained model exists, the app trains a small demo Random Forest model from the sample dataset.

## Evaluate a Dataset

```bash
python evaluate.py --data data/sample_urls.csv
```

The CSV must contain:

```csv
url,label
https://www.example.com,0
http://paypal.verify-account.example.test/login,1
```

Labels use `0 = legitimate` and `1 = phishing`.

## Evaluate on the UCI PhiUSIIL Real Dataset

Install requirements, then run:

```bash
python real_dataset_eval.py --max-rows 30000
```

This fetches UCI dataset `id=967`, converts its labels to the project convention, trains/evaluates the handcrafted-feature models, and trains a character n-gram URL text model. The UCI dataset uses `0 = phishing` and `1 = legitimate`; PhishGuard-Lite uses `0 = legitimate` and `1 = phishing`, so the script converts labels automatically.

To include the optional lightweight deep learning model:

```bash
python real_dataset_eval.py --max-rows 30000 --include-deep --deep-max-train 6000 --deep-epochs 1
```

Generated real-dataset results are saved to:

- `results/tables/uci_phiusiil_real_dataset_results.csv`
- `results/figures/uci_phiusiil_confusion_matrix_rf.png`
- `results/figures/uci_phiusiil_confusion_matrix_tfidf.png`

By default, the script does not save raw external URLs locally. Use `--save-url-sample` only if you intentionally want a local URL sample for further experimentation.

## Analyze One URL from the Command Line

```bash
python predict.py "https://paypal.verify-account.example-login.test/security/update"
```

Enable the optional Hugging Face layer:

```bash
python predict.py "https://paypal.verify-account.example-login.test/security/update" --use-hf
```

## Generate Report Assets

```bash
python generate_report_assets.py
```

Generated files include:

- `results/figures/confusion_matrix.png`
- `results/figures/feature_importance.png`
- `results/figures/feature_selection_comparison.png`
- `results/tables/model_comparison_table.csv`
- `results/tables/latency_comparison_table.csv`
- `report/figures/*.png`

## Project Structure

```text
phishguard-lite/
├── README.md
├── requirements.txt
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── real_dataset_eval.py
├── generate_report_assets.py
├── src/
├── data/
├── models/
├── results/
└── report/
```

## Hybrid Risk Score

When the Hugging Face layer is available:

```text
final_score = 0.40 * classical_ml_score + 0.40 * hf_score + 0.20 * rule_score
```

When the Hugging Face layer is disabled or unavailable:

```text
final_score = 0.70 * classical_ml_score + 0.30 * rule_score
```

Risk levels:

- `LOW`: final score below `0.40`
- `MEDIUM`: final score from `0.40` to below `0.70`
- `HIGH`: final score `0.70` or higher

Prediction:

- `phishing` if final score is at least `0.50`
- `legitimate` otherwise

## Hugging Face Fallback Behavior

The Hugging Face model is optional because pretrained models may require internet access, disk space, and more memory than a small classroom laptop can provide. If loading fails, PhishGuard-Lite continues running with:

- the trained classical ML model
- the rule-based risk layer
- the fallback hybrid formula

The Streamlit app keeps Hugging Face disabled by default for reliable offline demos.

## Troubleshooting

If `streamlit` is not found, install dependencies again:

```bash
pip install -r requirements.txt
```

If `models/phishguard_rf.joblib` is missing:

```bash
python train.py
```

If a Hugging Face model fails to load, leave the app setting disabled. The project is designed to remain usable without that layer.
