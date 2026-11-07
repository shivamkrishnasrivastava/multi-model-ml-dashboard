import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# 1. Load Dataset
# https://www.kaggle.com/datasets/guilhermegoat/heart-disease-health-indicators
df = pd.read_csv("heart_disease_health_indicators.csv")

if len(df) > 5000:
    df = df.sample(n=5000, random_state=42)

X = df.drop(columns=['HeartDiseaseorAttack'])
y = df['HeartDiseaseorAttack']

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_state=42, test_size=0.2, stratify=y
)

# Save test set for Streamlit testing
test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv("test_data.csv", index=False)

# 3. Preprocessing (Standard Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.joblib")