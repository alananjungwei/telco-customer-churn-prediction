# Telco Customer Churn Prediction

End-to-end customer churn analysis project combining machine learning, feature engineering, customer lifecycle analysis, and business-oriented behavioral insights to identify churh drivers and retention opportunities. 

# Business Problem

Customer churn is one of the most important business challenges in the telecommunications industry. Retaining existing customers is often significantly cheaper than acquiring new customers.

This project aims to:

- Predict customer churn using machine learning
- Identify major churn drivers
- Engineer behavioral customer segments
- Analyze customer lifecycle dynamics
- Compare multiple classification models and encoding strategies


---

# Key Business Finidngs

| Finding | Observation | Business Recommendation |
|---|---|---|
| Short-Term High Spenders | Newly acquired high-spending customers exhibited substantially elevated churn rates. | Improve onboarding experience, customer support, and early-stage retention offers for premium new customers. |
| Long-Term High-Value Customers | Long-term premium customers demonstrated significantly stronger retention stability. | Develop loyalty programs and long-term customer reward incentives. |
| Service Bundling | Customers subscribed to more telecom services generally showed lower churn tendencies. | Promote bundled service packages to strengthen ecosystem integration and reduce switching behavior. |
| Spending Intensity | Customers with high spending intensity relative to tenure (`MonthlyChargePerTenure`) were much more likely to churn early. | Prioritize retention targeting for high-spending early-stage customers before churn occurs. |

<p align="center">
  <img src="images/business_finding.png" width="100%">
</p>


---

# Dataset

Dataset source: [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Dataset Overview

* 7043 telecom customers
* Demographic information
* Service subscriptions
* Billing information
* Contract information

Target Variable:

* Churn (Yes / No)

---


# Project Workflow

```py
Data Cleaning
↓
EDA
↓
Encoding
↓
Model Comparison
↓
Hyperparameter Tuning
↓
Feature Engineering
↓
Business Insights
```

---

# Engineered Features

|Feature|Purpose|
|-------|-------|
|Total Services|Service ecosystem management|
|Long Term Customer|CUstomer lifecycle stability|
|Short Term High Spender|Early churn risk prediction|
|Loyal High Value Customer|Premium customer retention|
|Monthly Charge Per Tenure|Spending intensity measurement|

---

# Best Model Results

| Experiment | Best Model | Test Acc | ROC-AUC | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|---|
|Full Encoding|Logistic Regression|0.804|0.835|0.57|0.61|
|Drop-First Encoding|Logistic Regression|0.804|0.836|0.57|0.61|
|Feature Engineered|Gradient Boosting|0.800|0.841|0.53|0.58|
|Engineered Features Only|XGBoost|0.781|0.826|0.44|0.51|


---

# Additional Documentation

* Technical Report
* Business Report
* Jupyter Notebook

---

# Conclusion

Feature engineering provided valuable customer lifecycle insights while maintaining strong predictive performance. The project demonstrated that behavioral customer segments and spending intensity are important indicators of churn risk and can support targeted retention strategies. 




























