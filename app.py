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
            scaler = joblib.load("model/scaler.joblib")

            # Scale data if required
            if selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine", "Naive Bayes"]:
                X_proc = scaler.transform(X_test)
            else:
                X_proc = X_test

            # Inference (No Retraining)
            y_pred = model.predict(X_proc)
            
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_proc)[:, 1]
            else:
                y_proba = y_pred

            # Calculate Evaluation Metrics
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC Score": roc_auc_score(y_test, y_proba),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "MCC Score": matthews_corrcoef(y_test, y_pred)
            }

            st.markdown(f"### 📈 Performance Metrics for: `{selected_model_name}`")
            cols = st.columns(6)
            for col, (metric_name, val) in zip(cols, metrics.items()):
                col.metric(metric_name, f"{val:.4f}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()

            # Visualizations
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("🧩 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                            xticklabels=["Negative", "Positive"],
                            yticklabels=["Negative", "Positive"])
                ax.set_xlabel("Predicted Label", fontsize=10)
                ax.set_ylabel("True Label", fontsize=10)
                st.pyplot(fig)

            with col_right:
                st.subheader("📋 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0, color="#dbeafe"), use_container_width=True)

        except Exception as e:
            st.error(f"Error executing inference: {e}")


else:
    st.info("👈 Please upload your `test_data.csv` file from the sidebar to begin.")