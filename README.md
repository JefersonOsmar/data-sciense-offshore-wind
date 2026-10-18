# Data Science for Offshore Wind Energy

This repository contains interactive machine learning and time-series workflows designed for the analysis, technological classification, techno-economic assessment, and atmospheric forecasting of offshore wind energy systems. Utilizing spatial, physical, environmental, and financial data, the project implements robust data preprocessing, feature scaling, hyperparameter tuning, deep neural network training, and cross-validation within Jupyter Notebooks to solve classification, regression, and sequential forecasting tasks.

---

## 📌 Project Overview
The repository focuses on three primary engineering, economic, and atmospheric challenges in the offshore wind energy sector:

1. **Technological Foundation Classification (Classification):** Predicting the optimal turbine foundation type (*Fixed-bottom* vs. *Floating*) based on site-specific physical parameters like water depth and distance to the coast.
2. **Techno-Economic LCOE Estimation (Regression):** Predicting the All-In Levelized Cost of Energy (LCOE) for future horizons (e.g., 2035) using geographic, technical, and capacity factors.
3. **Wind Speed Time-Series Forecasting (Sequential):** Predicting historical monthly wind speeds across long horizons (36-month test windows) using advanced statistical models and recurrent deep learning architectures to support energy yield planning.

---

## 🛠️ Implemented Machine Learning & Time-Series Algorithms
To build a robust comparative framework, multiple paradigms of Machine Learning and Statistical Modeling are explored across the interactive pipelines:

### 🔹 1. Classification Notebook (Foundation Type Prediction)
*   **K-Nearest Neighbors (K-NN):** Optimized via `GridSearchCV` evaluating neighbor sizes, distance weights (*uniform* vs. *distance*), and metrics (*euclidean* vs. *manhattan*).
*   **Decision Tree Classifier:** Supervised tree structures mapping regional geotechnical boundaries.
*   **Random Forest Classifier:** Ensemble technique using bagging to achieve high generalization capability.
*   **Logistic Regression & MLP Classifier (Neural Networks):** Linear baselines and multi-layer perceptrons for complex boundary mapping.

### 🔸 2. Regression Notebook (LCOE Estimation)
*   **Polynomial Regression with Regularization:** Combines feature interaction mapping (`PolynomialFeatures`) with `ElasticNet` (Ridge/Lasso balance) to mitigate overfitting.
*   **K-Neighbors Regressor & Decision Tree Regressor:** Non-parametric approaches for localized cost trend mapping.
*   **Random Forest & MLP Regressor:** High-capacity non-linear architectures for global cost surface optimization.

### 🍃 4. Time-Series Forecasting Notebook (Wind Speed Prediction Pipeline)
*   **Stationarity & Seasonal Adjustments:** Implements automated 12-month seasonal differencing to isolate and stabilize long-term environmental trends.
*   **M1: ARMA(2,0,2):** Classical linear autoregressive moving-average baseline modeled on the seasonally differenced wind series.
*   **M2: ARIMA(2,1,2):** Classical integrated time-series framework evaluating dynamic trend components.
*   **M3: Deep LSTM (Recurrent Neural Network):** High-capacity deep learning model structured with stacked LSTM layers (64 units, `tanh` activation), `Dropout(0.2)` regularizers to prevent overfitting, and an optimized `Dense` layer framework.

---

## 📁 Repository Structure
```text
├── data/
│   ├── open_supply-curve_post_proc.csv   # Processed offshore wind supply curve dataset
│   └── albacora_wind_30anos.csv         # 30-year historical wind speed time-series dataset
├── notebooks/
│   ├── 1_foundation_classification.ipynb # Exploratory analysis & multi-algorithm classification pipeline
│   ├── 2_lcoe_regression.ipynb           # Feature correlations & multi-algorithm LCOE regression pipeline
│   └── 3_wind_speed_forecasting.ipynb    # Time-series pipeline (ARMA, ARIMA, and Deep LSTM)
├── requirements.txt                      # Core dependencies (pandas, tensorflow, scikit-learn, statsmodels, etc.)
└── README.md                             # Technical documentation
```

---

## 📊 Key Features & Interactive Pipeline
*   **Feature Engineering & Exploration:** Computes statistical distributions, categorical encoding (`cat.codes`), visual Pearson correlation matrices, and time-series seasonal decompositions (`seasonal_decompose`).
*   **Data Standardization & Windowing:** Implements `StandardScaler` for tabular structures and a dynamic `MinMaxScaler(feature_range=(-1, 1))` sliding window helper function (`look_back=12`) for sequential LSTM feeding to block data leakage.
*   **Model Optimization:** Automated hyperparameter optimization using exhaustive search patterns under the `GridSearchCV` framework alongside adaptive neural network callbacks.
*   **Comprehensive Metrics Evaluation:**
    *   *Classification:* Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
    *   *Regression & Time Series:* Mean Absolute Error (MAE), Coefficient of Determination (\(R^2\)), and Root Mean Squared Error (RMSE) mapped back onto the actual physical scale (\(m/s\)).

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
4. Open the notebooks inside the `notebooks/` directory and run the cells sequentially.

---

## 🛠️ Main Tech Stack
*   **Language:** Python 3.13
*   **Core Libraries:** TensorFlow 2.x, scikit-learn, statsmodels, pandas, numpy, matplotlib, seaborn, notebook
