# Customer Churn Prediction (ANN)

A small end-to-end machine learning demo: a **Keras artificial neural network** predicts whether a bank customer is likely to **churn**, with a **Streamlit** web UI for interactive inputs.

This repo pairs experimentation notebooks with a deployable app that loads a saved model and the same preprocessing objects used at training time.

## What it does

- Collects customer attributes (credit score, geography, gender, age, tenure, balance, products, card/membership flags, estimated salary).
- Applies **one-hot encoding** for geography, **label encoding** for gender, and **standard scaling** on the numeric feature vector (order matches the fitted `StandardScaler`).
- Runs inference with `my_model.h5` and shows churn vs. no-churn plus a **probability** (threshold at 0.5).

## Tech stack

| Area | Libraries |
|------|-----------|
| Model | TensorFlow / Keras |
| Preprocessing | scikit-learn, pandas, numpy |
| App | Streamlit |

## Repository layout

| Path | Role |
|------|------|
| `app.py` | Streamlit churn prediction UI |
| `my_model.h5` | Trained Keras model (must be present to run the app) |
| `label_encoder_gender.pkl` | Fitted label encoder for gender |
| `one_hot_encoder_geo.pkl` | Fitted one-hot encoder for geography |
| `scaler.pkl` | Fitted `StandardScaler` for feature scaling |
| `experiments.ipynb` | Training / exploration workflow |
| `prediction.ipynb` | Prediction examples aligned with the app |
| `requirements.txt` | Python dependencies |

## Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended for TensorFlow compatibility on your platform).
- The four artifacts above (`my_model.h5` and the three `.pkl` files) in the **same directory** as `app.py` when you launch Streamlit.

## Setup

1. Clone or copy this project and open a terminal in the project root.

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   `pickle` is part of the Python standard library; you do not need to install it via pip. If `pip` complains about `pickle` in `requirements.txt`, remove that line locally or install everything except that entry.

## Run the app

From the project root (where `app.py` and the model/encoder files live):

```bash
streamlit run app.py
```

The browser opens the **Customer Churn Prediction** page. Adjust inputs; the app prints whether the customer is likely to churn and the raw probability.

## Retraining or swapping the model

1. Use `experiments.ipynb` (or your own training script) to fit a new model and preprocessors on your dataset.
2. Save the Keras model (e.g. `model.save("my_model.h5")` or the Keras 3 equivalent you use).
3. Persist the same encoders and scaler with `pickle` so column order and categories stay consistent with `app.py`.
4. Replace the files in the project root and restart Streamlit.

## License

Add a license file if you publish this repo (e.g. MIT). Course/tutorial projects often omit one; choose what fits your use case.
