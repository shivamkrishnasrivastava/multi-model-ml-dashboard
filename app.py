import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Heart Disease ML Evaluation", layout="wide")

st.title("📊 Multi-Model Classification Dashboard")
st.write("Upload test dataset (CSV) and select a trained ML model to evaluate predictions.")

# Model File Map
MODEL_MAP = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

# Sidebar Controls
st.sidebar.header("Configuration")
selected_model_name = st.sidebar.selectbox("Select ML Model", list(MODEL_MAP.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test CSV (`test_data.csv`)", type=["csv"])

target_col = st.sidebar.text_input("Target Column Name", value="HeartDiseaseorAttack")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    with st.expander("📌 View Uploaded Test Data Preview", expanded=False):
        st.dataframe(test_df.head(), use_container_width=True)

    if target_col not in test_df.columns:
        st.error(f"⚠️ Target column '{target_col}' not found in uploaded dataset.")
    else:
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # Load Model and Scaler
        try:
            model = joblib.load(MODEL_MAP[selected_model_name])

        except Exception as e:
            st.error(f"Error executing inference: {e}")


else:
    st.info("👈 Please upload your `test_data.csv` file from the sidebar to begin.")