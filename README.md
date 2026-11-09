# AI-Based-Predictive-Maintenance-For-Induction-Motor
**Objective**

The objective of this project is to predict the Remaining Useful Life (RUL) of an induction motor using machine learning techniques.

The project analyzes motor operating parameters such as temperature, speed, frequency, voltage, current, power, and thermal load to estimate the motor's remaining operating life.

It also incorporates Explainable AI (XAI) using SHAP to identify the key factors influencing predictions and generate engineering-based maintenance recommendations.

**About the Dataset**

The project uses an induction motor operating dataset (induction_OTG_motor.csv) containing 11,500 observations and 11 features.

The dataset includes the following parameters:

Electrical Parameters: DC Bus Voltage, Frequency, Output Voltage, Output Current, and High Resolution Output Current.

Mechanical Parameters: Speed and Load Index.

Thermal Parameters: Temperature and Thermal Load.

Power Parameter: Power consumption.

Target Variable: Remaining Useful Life (RUL_hours).

**Method**

The project consists of the following main steps:

Data Loading and Exploration: Loading the motor dataset and examining its statistical properties, distributions, and correlations.

Data Preprocessing: Checking missing values, duplicate records, and outliers.

Feature Preparation: Separating input features and the target variable (RUL_hours).

Feature Scaling: Applying StandardScaler to standardize input features for Linear Regression.

Model Building and Training: Implementing Linear Regression, Random Forest Regressor, and XGBoost Regressor.

Hyperparameter Tuning: Applying RandomizedSearchCV with 5-fold cross-validation to optimize XGBoost performance.

Model Evaluation: Comparing models using MAE, RMSE, and R² score.

Explainable AI: Using SHAP to interpret model predictions and identify the most influential motor operating parameters.

Predictive Maintenance Application: Developing a Streamlit application to predict RUL, classify motor health, visualize SHAP explanations, and generate engineering maintenance recommendations.

**Requirements**

To run the project, install the following libraries:

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

XGBoost

SHAP

Joblib

Streamlit

Install using:

pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap joblib streamlit

**Motivation**

Unexpected induction motor failures can result in equipment damage, production downtime, and increased maintenance costs.

This project aims to:

Estimate the remaining useful life of an induction motor.

Identify important parameters affecting motor health.

Detect potential degradation before motor failure.

Interpret predictions using Explainable AI.

Generate engineering-based maintenance recommendations.

Support condition-based and predictive maintenance decisions.

**Models Used**

Machine Learning Models:

Linear Regression

Random Forest Regressor

XGBoost Regressor

Explainable AI:

SHAP (SHapley Additive exPlanations)

Application Development:

Streamlit

**Results**

The models were evaluated using an 80:20 training-testing split.

Model

MAE

RMSE

R² Score

Linear Regression

1642.11

2039.18

0.9061

Random Forest

1081.68

1556.53

0.9453

XGBoost

1078.54

1549.36

0.9458

Tuned XGBoost

1067.58

1520.56

0.9478

The tuned XGBoost model achieved the best performance, with an R² score of approximately 94.78%.

Predictive Maintenance Application

The Streamlit application provides:

User input for motor operating parameters.

Predicted Remaining Useful Life (RUL) in hours.

Motor health classification: Healthy, Good, Warning, Critical, and Failure Imminent.

Top three influential features using SHAP.

AI-based interpretation of feature contributions.

Engineering-based maintenance recommendations.

SHAP Waterfall Plot for prediction explainability.
