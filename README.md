# Heart Disease Prediction & Multi-Model Classifier Evaluation

## a. Problem Statement
Cardiovascular diseases are among the leading causes of death worldwide. Early identification and risk stratification of individuals at risk of heart disease enable proactive medical and lifestyle interventions. The objective of this project is to implement, evaluate, and compare five distinct machine learning classification models to predict the presence of heart disease or attack using comprehensive health and demographic indicators.

---

## b. Dataset Description
* **Dataset Name:** Heart Disease Health Indicators (Kaggle / UCI Machine Learning Repository)
* **Problem Type:** Binary Classification (Target Variable: `HeartDiseaseorAttack`: 0 = No, 1 = Yes)
* **Instance Size:** 5,000 instances (sampled from the full 253k+ dataset to optimize repository storage and model deployment)
* **Feature Size:** 21 input features (exceeding the minimum requirement of 12 features)
* **Key Features Include:** High Blood Pressure, High Cholesterol, BMI, Smoker Status, Stroke History, Diabetes, Physical Activity, General Health, Mental Health, Physical Health, Difficulty Walking, Sex, Age, Education, and Income.

---
## c. GitHub Repository Link
👉 [https://github.com/shivamkrishnasrivastava/multi-model-ml-dashboard]
---

---
## d. Streamlit Link
👉 [https://multi-model-ml-dashboard-wmocwgj8cwtwxchgstmhce.streamlit.app/]
---

## e. Models Used & Comparison

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9090 | 0.8586 | 0.6552 | 0.1900 | 0.2946 | 0.3198 |
| **Decision Tree** | 0.8970 | 0.7800 | 0.4348 | 0.1000 | 0.1626 | 0.1712 |
| **KNN** | 0.8820 | 0.7599 | 0.2632 | 0.1000 | 0.1449 | 0.1081 |
| **Naive Bayes** | 0.8130 | 0.8084 | 0.2919 | 0.6100 | 0.3948 | 0.3287 |
| **Support Vector Machine** | 0.9000 | 0.7280 | 0.5000 | 0.0600 | 0.1071 | 0.1469 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest **Accuracy (0.9090)**, **AUC (0.8586)**, and **Precision (0.6552)**. It effectively separates classes using linear decision boundaries with strong overall discrimination power. |
| **Decision Tree** | High overall accuracy (0.8970), but suffers from low Recall (0.1000) and low F1 score, indicating strong bias toward predicting the majority class (non-disease cases). |
| **KNN** | Recorded the lowest MCC score (0.1081) and poor recall (0.1000). Distance calculations struggled to effectively separate positive cases in the high-dimensional feature space. |
| **Naive Bayes** | Outperformed all other models in **Recall (0.6100)** and **F1 Score (0.3948)**, successfully catching the majority of positive heart disease cases despite lower precision due to independence assumptions. |
| **Support Vector Machine** | Maintained high overall accuracy (0.9000), but demonstrated severely poor Recall (0.0600) and low AUC (0.7280) due to decision boundary bias toward the majority class without probability calibration/class weighting. |

### Overall, Winner for your dataset?
**Logistic Regression** is the clear winner for this dataset, achieving the top score in **Accuracy (0.9090)**, **AUC (0.8586)**, and **Precision (0.6552)**, making it the most balanced and reliable overall classifier. *(Note: Naive Bayes is the best candidate if clinical priority is maximizing sensitivity/recall).*

## Project Repository Structure
```text
multi-model-ml-dashboard/
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   └── scaler.joblib
│
├── app.py
├── train.py
├── test_data.csv
├── requirements.txt
└── README.md
```

---

## Required Dependencies (`requirements.txt`)
The project relies on the following Python packages:
* `streamlit`
* `scikit-learn`
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `joblib`

---

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/shivamkrishnasrivastava/multi-model-ml-dashboard.git
   cd multi-model-ml-dashboard
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```