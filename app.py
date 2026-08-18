from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
RESULTS_FILE = BASE_DIR / "results" / "model_metrics.csv"
TARGET = "diagnosis"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "Support Vector Machine (Additional 6th)": "svm.joblib",
}

st.set_page_config(
    page_title="ML Assignment 2 - Breast Cancer Classification",
    page_icon="📊",
    layout="wide",
)

st.title("Machine Learning Assignment 2")
st.subheader("Breast Cancer Wisconsin (Diagnostic) — Classification Model Evaluation")
st.caption(
    "Academic demonstration only. This application is not a medical diagnostic tool."
)

with st.expander("Dataset and target definition"):
    st.write(
        "The project uses 30 numeric predictors from the Breast Cancer Wisconsin "
        "(Diagnostic) dataset. The target column is `diagnosis`: "
        "`1 = malignant`, `0 = benign`."
    )

metrics_reference = pd.read_csv(RESULTS_FILE)
st.markdown("### Training-time model comparison")
st.dataframe(
    metrics_reference.style.format({
        "Accuracy": "{:.4f}",
        "AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "MCC": "{:.4f}",
    }),
    use_container_width=True,
)

uploaded = st.file_uploader(
    "Upload test data (CSV)",
    type=["csv"],
    help="Use the supplied test_data.csv. It must contain the `diagnosis` target column.",
)

selected_model = st.selectbox("Select a classification model", list(MODEL_FILES))

if uploaded is None:
    st.info("Upload `test_data.csv` to evaluate the selected model.")
    st.stop()

df = pd.read_csv(uploaded)

if TARGET not in df.columns:
    st.error("The uploaded CSV must contain a `diagnosis` column.")
    st.stop()

X = df.drop(columns=[TARGET])
y_true = df[TARGET]

model = joblib.load(MODEL_DIR / MODEL_FILES[selected_model])

try:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
except Exception as exc:
    st.error(
        "The uploaded columns do not match the model features. "
        f"Details: {exc}"
    )
    st.stop()

accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)

st.markdown(f"### Evaluation — {selected_model}")

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("AUC", f"{auc:.4f}")
c3.metric("Precision", f"{precision:.4f}")

c4, c5, c6 = st.columns(3)
c4.metric("Recall", f"{recall:.4f}")
c5.metric("F1 Score", f"{f1:.4f}")
c6.metric("MCC", f"{mcc:.4f}")

left, right = st.columns(2)

with left:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign (0)", "Malignant (1)"],
    ).plot(ax=ax, cmap="Greys", colorbar=False)
    ax.set_title(f"{selected_model} — Confusion Matrix")
    st.pyplot(fig, clear_figure=True)

with right:
    st.markdown("#### Classification Report")
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.round(4), use_container_width=True)

st.markdown("#### Prediction preview")
preview = df.copy()
preview["predicted_diagnosis"] = y_pred
preview["malignant_probability"] = y_prob
st.dataframe(preview.head(30), use_container_width=True)

st.download_button(
    "Download predictions as CSV",
    data=preview.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)
