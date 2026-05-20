# telco-customer-churn-prediction
End-to-end machine learning project for predicting telecom customer churn using Logistic Regression, Random Forest, AdaBoost, Gradient Boosting, and XGBoost with hyperparameter tuning, feature engineering, and model interpretability analysis.
# Data is from kaggle... 

# Answering business questions


# Data Visualization


# Exploratory Data Analysis (EDA)


# Train-test split


# Machine Learning


# Model Performance Summary 

## Main Results Table 

| Experiment | Best Model | Test Acc | ROC-AUC | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|---|
|Full Encoding|Logistic Regression|0.804|0.835|0.57|0.61|
|Drop-First Encoding|Logistic Regression|0.804|0.836|0.57|0.61|
|Feature Engineered|Gradient Boosting|0.800|0.841|0.53|0.58|
|Engineered Features Only|XGBoost|0.781|0.826|0.44|0.51|



### Full One-Hot Encoding (`drop_first=False`)

| Model | Train Acc | Test Acc | ROC-AUC | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.804 | 0.835 | 0.65 | 0.57 | 0.61 |
| KNN | 0.838 | 0.743 | 0.769 | 0.52 | 0.52 | 0.52 |
| XGBoost | 0.822 | 0.798 | 0.839 | 0.64 | 0.54 | 0.59 |
| AdaBoost | 0.807 | 0.793 | 0.836 | 0.63 | 0.54 | 0.58 |
| Gradient Boosting | 0.819 | 0.796 | 0.839 | 0.64 | 0.53 | 0.58 |
| Random Forest | 0.856 | 0.795 | 0.833 | 0.64 | 0.52 | 0.58 |

### Reduced One-Hot Encoding (`drop_first=True`)

| Model | Train Acc | Test Acc | ROC-AUC | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.804 | 0.836 | 0.65 | 0.57 | 0.61 |
| KNN | 0.835 | 0.754 | 0.767 | 0.54 | 0.54 | 0.54 |
| XGBoost | 0.827 | 0.792 | 0.838 | 0.63 | 0.53 | 0.58 |
| AdaBoost | 0.808 | 0.792 | 0.842 | 0.64 | 0.51 | 0.57 |
| Gradient Boosting | 0.817 | 0.794 | 0.842 | 0.64 | 0.52 | 0.57 |
| Random Forest | 0.866 | 0.790 | 0.833 | 0.63 | 0.51 | 0.56 |

### Feature Engineered

| Model | Train Acc | Test Acc | ROC-AUC | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.813 | 0.799 | 0.839 | 0.64 | 0.55 | 0.59 |
| KNN | 0.833 | 0.768 | 0.772 | 0.57 | 0.55 | 0.56 |
| XGBoost | 0.829 | 0.798 | 0.839 | 0.64 | 0.54 | 0.59 |
| AdaBoost | 0.807 | 0.796 | 0.838 | 0.64 | 0.53 | 0.58 |
| Gradient Boosting | 0.820 | 0.800 | 0.841 | 0.65 | 0.53 | 0.58 |
| Random Forest | 0.873 | 0.788 | 0.832 | 0.63 | 0.49 | 0.55 |

### Engineered features only
| Model | Train Acc | Test Acc | ROC-AUC | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.787 | 0.778 | 0.806 | 0.64 | 0.38 | 0.48 |
| KNN | 0.731 | 0.768 | 0.745 | 0.58 | 0.44 | 0.50 |
| XGBoost | 0.801 | 0.781 | 0.826 | 0.63 | 0.44 | 0.51 |
| AdaBoost | 0.797 | 0.775 | 0.813 | 0.59 | 0.51 | 0.54 |
| Gradient Boosting | 0.817 | 0.794 | 0.842 | 0.64 | 0.52 | 0.57 |
| Random Forest | 0.843 | 0.775 | 0.808 | 0.60 | 0.45 | 0.51 |


# Conclusion 