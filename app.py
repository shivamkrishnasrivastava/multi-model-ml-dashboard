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

# Page Configuration
st.set_page_config(
    page_title="Multi-Model ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Light Mode CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }

    /* Modern Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.2);
    }
    .hero-banner h1 {
        color: white !important;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    .hero-banner p {
        color: #e2e8f0;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Custom Metric Cards */
    .custom-metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
        text-align: left;
    }
    .custom-metric-card:hover {
        transform: translateY(-3px);
        border-color: #93c5fd;
        box-shadow: 0 10px 20px -5px rgba(37, 99, 235, 0.1);
    }
    .metric-title {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Card Containers */
    .dashboard-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }

    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="status-badge">⚡ SYSTEM ONLINE</div>
    <h1>Multi-Model Classification Dashboard</h1>
    <p>Evaluate and benchmark machine learning algorithms with production-grade performance metrics and visual telemetry.</p>
</div>
""", unsafe_allow_html=True)

# Model File Map
MODEL_MAP = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

# Sidebar Configuration
st.sidebar.markdown("### 🎛️ Control Center")
selected_model_name = st.sidebar.selectbox("Choose Classifier", list(MODEL_MAP.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test CSV (`test_data.csv`)", type=["csv"])
target_col = st.sidebar.text_input("Target Column Name", value="HeartDiseaseorAttack")

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>Built with Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    with st.expander("📁 Preview Uploaded Dataset Structure", expanded=False):
        st.dataframe(test_df.head(), use_container_width=True)

    if target_col not in test_df.columns:
        st.error(f"⚠️ Target column '{target_col}' missing from uploaded dataset.")
    else:
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        try:
            model = joblib.load(MODEL_MAP[selected_model_name])
            scaler = joblib.load("model/scaler.joblib")

            if selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine", "Naive Bayes"]:
                X_proc = scaler.transform(X_test)
            else:
                X_proc = X_test

            y_pred = model.predict(X_proc)
            y_proba = model.predict_proba(X_proc)[:, 1] if hasattr(model, "predict_proba") else y_pred

            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC Score": roc_auc_score(y_test, y_proba),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "MCC Score": matthews_corrcoef(y_test, y_pred)
            }

            st.markdown(f"### 📈 Performance Metrics: <span style='color: #2563eb;'>{selected_model_name}</span>", unsafe_allow_html=True)
            
            # Render Custom HTML Metric Cards
            cols = st.columns(6)
            for col, (metric_name, val) in zip(cols, metrics.items()):
                with col:
                    st.markdown(f"""
                    <div class="custom-metric-card">
                        <div class="metric-title">{metric_name}</div>
                        <div class="metric-value">{val:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Visualizations Layout with Tabs & Clean Containers
            tab1, tab2 = st.tabs(["🧩 Confusion Matrix Analysis", "📋 Classification Report Breakdown"])

            with tab1:
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                col_m1, col_m2 = st.columns([1, 1])
                with col_m1:
                    st.markdown("#### Visual Heatmap")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    fig.patch.set_facecolor('#ffffff')
                    ax.set_facecolor('#ffffff')
                    
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                                xticklabels=["Negative", "Positive"],
                                yticklabels=["Negative", "Positive"],
                                annot_kws={"size": 12, "weight": "bold"})
                    
                    ax.tick_params(colors='#475569', labelsize=10)
                    ax.xaxis.label.set_color('#475569')
                    ax.yaxis.label.set_color('#475569')
                    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
                    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
                    st.pyplot(fig)
                with col_m2:
                    st.markdown("#### Matrix Insights")
                    st.markdown(f"""
                    * **True Negatives:** {cm[0,0]}
                    * **False Positives:** {cm[0,1]}
                    * **False Negatives:** {cm[1,0]}
                    * **True Positives:** {cm[1,1]}
                    
                    The confusion matrix illustrates the breakdown of correct vs incorrect classifications for **{selected_model_name}** on the provided test dataset.
                    """)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                st.markdown("#### Detailed Classification Report Table")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0, color="#dbeafe"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error executing inference pipeline: {e}")
else:
    st.markdown("""
    <div class="dashboard-container" style="text-align: center; padding: 4rem 2rem;">
        <h3>👈 Awaiting Test Data</h3>
        <p style="color: #64748b; max-width: 500px; margin: 0 auto;">Please upload your <code>test_data.csv</code> file using the sidebar control panel to initiate evaluation and render model metrics.</p>
    </div>
    """, unsafe_allow_html=True)