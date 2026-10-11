# Data Science for Offshore Wind Energy

This repository contains interactive machine learning workflows designed for the analysis, technological classification, and techno-economic assessment of offshore wind energy systems. Utilizing spatial, physical, and economic data, the project implements robust data preprocessing, feature scaling, hyperparameter tuning, and cross-validation within Jupyter Notebooks to solve both classification and regression tasks.

## 📌 Project Overview

The repository focuses on two primary engineering and economic problems in the offshore wind sector:
1. **Technological Foundation Classification (Classification):** Predicting the optimal turbine foundation type (Fixed-bottom vs. Floating) based on site-specific physical parameters like water depth and distance to the coast.
2. **Techno-Economic LCOE Estimation (Regression):** Predicting the All-In Levelized Cost of Energy (LCOE) for future horizons (e.g., 2035) using geographic, technical, and capacity factors.

---

## 🛠️ Implemented Machine Learning Algorithms

To build a robust comparative framework, multiple paradigms of Machine Learning are explored across the interactive pipelines:

### 🔹 Classification Notebook (Foundation Type Prediction)
* **K-Nearest Neighbors (K-NN):** Optimized via `GridSearchCV` evaluating neighbor sizes, distance weights (`uniform` vs. `distance`), and metrics (`euclidean` vs. `manhattan`).
* **Decision Tree Classifier:** Supervised tree structures mapping regional boundaries.
* **Random Forest Classifier:** Ensemble technique using bagging to achieve high generalization capability.
* **Logistic Regression & MLP Classifier (Neural Networks):** Linear baselines and multi-layer perceptrons for complex boundary mapping.
* *Robustness:* Evaluated using **Stratified K-Fold Cross-Validation** to safely manage any inherent target variable imbalances.

### 🔸 Regression Notebook (LCOE Estimation)
* **Polynomial Regression with Regularization:** Combines feature interaction mapping (`PolynomialFeatures`) with `ElasticNet` (Ridge/Lasso balance) to mitigate overfitting.
* **K-Neighbors Regressor & Decision Tree Regressor:** Non-parametric approaches for localized cost trend mapping.
* **Random Forest & MLP Regressor:** High-capacity non-linear architectures for global cost surface optimization.

---

## 📁 Repository Structure

```text
├── data/
│   └── open_supply-curve_post_proc.csv   # Processed offshore wind supply curve dataset
├── notebooks/
│   ├── 1_foundation_classification.ipynb # Exploratory analysis & multi-algorithm classification pipeline
│   └── 2_lcoe_regression.ipynb           # Feature correlations & multi-algorithm LCOE regression pipeline
├── requirements.txt                      # Core dependencies (pandas, scikit-learn, seaborn, etc.)
└── README.md                             # Technical documentation
```

---

## 📊 Key Features & Interactive Pipeline

* **Feature Engineering & Exploration:** Computes statistical distributions, categorical encoding (`cat.codes`), and evaluates multi-collinearity via visual Pearson correlation matrices.
* **Data Standardization:** Implements `StandardScaler` inside localized pipeline structures to enforce a strict boundary against *data leakage* between training and testing sets.
* **Model Optimization:** Automated hyperparameter optimization using exhaustive search patterns under the `GridSearchCV` framework.
* **Comprehensive Metrics Evaluation:**
  * *Classification:* Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
  * *Regression:* Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination (\(R^2\)).

---

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd data-sciense-offshore-wind
   ```

2. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Lab or Jupyter Notebook:**
   ```bash
   jupyter lab
   ```
   Open the notebooks inside the `notebooks/` directory to run the cells sequentially.

## 🛠️ Main Tech Stack
* **Language:** Python 3.13
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `notebook`
