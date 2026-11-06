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