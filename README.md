# Machine Learning Capstone Project
### Course: 23CSE301 Machine Learning — B.Tech. Computer Science and Engineering
---

## Executive Summary

This repository houses the complete, end-to-end Machine Learning pipeline across **three tracks**:
1. **Regression Track:** Predicting crop yield across Indian agricultural states based on spatial, seasonal, and resource inputs.
2. **Classification Track:** Biometric identification of mushroom edibility (*Edible* vs *Poisonous*) using morphological attributes.
3. **Clustering Track:** Unsupervised market segmentation of wholesale clients across 6 core purchasing categories.

The project strictly follows the **23CSE301 Capstone Guidelines and Evaluation Rubrics**, fulfilling **all 22 required algorithms** across both Review 1 and Review 2, complete with hyperparameter tuning, cross-validation, and diagnostic visualisations.

In addition, an **interactive web-based graphical interface (GUI)** built with **Streamlit, Plotly, and a Soft Pastel Glassmorphism design system** is provided alongside **cloud deployment configurations (+2 Bonus Marks)**.

---

## Comprehensive Results Summary & Benchmark Tables

### 1. Regression Track: Crop Yield Prediction
- **Target:** `yield` (tonnes per hectare)
- **Held-out Split:** 80% Train, 20% Test (`random_state=42`)
- **Engineered Features:** `fertilizer_per_area = fertilizer / area`, `pesticide_per_area = pesticide / area`

| Rank | Algorithm | Test $R^2$ Score | RMSE (tonnes/ha) | MAE (tonnes/ha) | 5-Fold CV $R^2$ (Mean ± Std) | Hyperparameter Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **Polynomial Regression** | **0.9900** | **89.4538** | 9.5657 | **0.9576 ± 0.0429** | Degree=2, OLS Fit |
| 2 | **Random Forest Regressor** | **0.9890** | **93.8497** | **7.2289** | **0.9575 ± 0.0575** | `max_depth=20, n_est=100, split=2` |
| 3 | **Gradient Boosting Regressor**| 0.9881 | 97.6690 | 9.1121 | — | `learning_rate=0.1, depth=5, n_est=200` |
| 4 | **Decision Tree Regressor** | 0.9754 | 140.4837 | 10.0405 | — | `max_depth=10` |
| 5 | **K-Nearest Neighbors Regressor**| 0.9109 | 267.2388 | 17.1790 | — | `k=5`, Euclidean metric |
| 6 | **ElasticNet Regression** | 0.8075 | 392.7028 | 67.6115 | — | $\alpha=0.001, l_1=0.5$ |
| 7 | **Ridge Regression** | 0.8035 | 396.8156 | 63.4070 | — | $\alpha=1.0$ ($L_2$ Regularization) |
| 8 | **Lasso Regression** | 0.8023 | 398.0417 | 62.9032 | — | $\alpha=0.001$ ($L_1$ Sparsity) |
| 9 | **Linear Regression (Baseline)**| 0.8023 | 398.0430 | 62.9366 | — | Ordinary Least Squares |
| 10 | **Support Vector Regressor (SVR)**| 0.0140 | 888.8164 | 75.5081 | — | RBF Kernel, Scaled features |

---

### 2. Classification Track: Mushroom Bio-Toxicity
- **Target:** `class` (0: Edible `e`, 1: Poisonous `p`)
- **Held-out Split:** 80% Train, 20% Test stratified (`random_state=42`)
- **Engineered Feature:** `stalk-color-match = (stalk-color-above-ring == stalk-color-below-ring)`

| Review Part | Algorithm | Test Accuracy | Precision | Recall | Weighted F1 | ROC-AUC (OvR) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Part A** | **Logistic Regression** | 0.998769 | 1.000000 | 0.997446 | 0.998769 | 1.000000 |
| **Part A** | **K-Nearest Neighbors ($k=3$)** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |
| **Part A** | **Gaussian Naive Bayes** | 0.964923 | 0.934211 | 0.997446 | 0.964927 | 0.996881 |
| **Part A** | **Decision Tree Classifier** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |
| **Part A** | **Support Vector Machine (SVC)** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |
| **Part B** | **Random Forest Classifier** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |
| **Part B** | **AdaBoost Classifier** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |
| **Part B** | **Gradient Boosting Classifier** | 0.998769 | 1.000000 | 0.997446 | 0.998769 | 1.000000 |
| **Part B** | **Bagging Classifier (Tree Base)**| **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |
| **Part B** | **MLP Classifier (Neural Net)** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |

- **Top-2 5-Fold Cross Validation:**
  - Random Forest: Mean CV Accuracy = **1.000000** (Folds: `[1.0, 1.0, 1.0, 1.0, 1.0]`)
  - K-Nearest Neighbors: Mean CV Accuracy = **0.999692** (Folds: `[0.998462, 1.0, 1.0, 1.0, 1.0]`)

---

### 3. 🛒 Clustering Track: Wholesale Customer Segmentation
- **Features (6 Channels):** `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen`
- **Total Clients:** 440 wholesale observations
- **Elbow Point:** $k=5$ (Inertia elbow reduction slope plateau)

| Algorithm | Configuration | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) | Cluster Distribution |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **K-Means Clustering** | $k=5$, Euclidean | **0.3690** | **0.9125** | **162.4134** | $C_0: 270, C_1: 10, C_2: 63, C_3: 96, C_4: 1$ |
| **Agglomerative Hierarchical** | $k=4$, Ward Linkage | 0.2670 | 0.9245 | 120.1315 | $C_0: 281, C_1: 5, C_2: 153, C_3: 1$ |
| **K-Means (log1p Transformed)** | $k=5$, Log + Scaler | 0.1916 | 1.4843 | 118.0005 | Comparison baseline |


All metrics and figures displayed in the GUI platform are generated directly from the identical data splits, random seeds (`random_state=42`), and preprocessing pipelines demonstrated in the submission notebooks.
Zero metrics are fabricated or hallucinated.

