# 🤖 MACHINE-LEARNING-TASK  
### 🧩 Two End-to-End Machine Learning Projects: Manufacturing Efficiency & Retail Web Session Intelligence  

---

## 🗂️ Repository Structure
```
MACHINE-LEARNING-TASK/
│
├── task1/
│ ├── manufacturing.ipynb
│ └── manufacturing_data(in).csv
│
└── task2/
├── rwsi.ipynb
└── rwsi_data(in).csv
```


Each task is an independent ML project covering complete stages — from **data preprocessing** to **model building**, **evaluation**, and **visualization**.

---

# 🏭 Task 1: Manufacturing Efficiency Analysis & Prediction  

---

## 🚀 Overview
This task predicts **manufacturing team efficiency** using workforce and production parameters.  
It explores how factors like **idle time**, **overtime**, **bonuses**, and **planned efficiency** influence the overall **performance score**.

---

## 🧩 Dataset Summary
- **File:** `manufacturing_data(in).csv`
- **Goal:** Predict team `efficiencyScore`
- **Records:** Several hundred production logs
- **Features:**  
  - 🧮 **Numerical:** plannedEfficiency', 'standardMinuteValue', 'workInProgress','overtimeMinutes', 'performanceBonus', 'idleMinutes',                          'idleWorkers','workerCount', 'efficiencyScore 
  - 🏷️ **Categorical:** 'recordDate', 'fiscalQuarter', 'productionDept', 'dayOfWeek', 'team', 'styleChangeCount' 

---

## 🔍 Exploratory Data Analysis (EDA)

### Univariate Analysis
- Most numeric features show **right skewness** (e.g., `idleMinutes`, `performanceBonus`).
- Outliers detected and treated using **IQR**.
- **Categorical Insights:**  
  - Certain departments maintain consistently higher efficiency.  
  - Higher **style changes** reduce performance slightly.  

### Bivariate Analysis
- **High Efficiency:** Low idle time, high planned efficiency, good bonuses.  
- **Negative correlation:** `idleMinutes` vs `efficiencyScore`.  
- **Positive correlation:** `plannedEfficiency` vs `efficiencyScore`.

---

## 🧹 Data Preprocessing
- **Outliers:** Percentile clipping and Percentile replacement  
- **Encoding:** OHE Encoding for categorical variables  
- **Scaling:** MinMaxScaler  
- **Data Split:** 80% Train | 20% Test  

---

## ⚙️ Model Building
Three regression models implemented:

| Model | Purpose |
|--------|----------|
| Linear Regression | Baseline model |
| Decision Tree Regressor | Handles nonlinear relationships |
| Random Forest Regressor | Ensemble model for best accuracy |

---

## 📊 Evaluation Metrics
| Metric | Description |
|--------|-------------|
| R² Score | Measures goodness of fit |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |

---

## 🧠 Model Comparison
| Model | R² Score | RMSE | Observation |
|--------|-----------|------|-------------|
| Linear Regression | ~0.34,0.77 | Moderate | Simple baseline |
| Decision Tree | ~0.27,0.48 | Lower | Captures nonlinearity |
| Random Forest | **~0.45,0.78** | **Lowest** | Best generalization |

✅ **Random Forest** achieved the best results with stable performance.

---

## 📈 Visualization Highlights
- **Heatmap:** Correlations among numeric features  
- **Boxplots:** Outlier detection  
- **Scatterplots:** Planned vs Actual efficiency  
- **Feature Importance:** plannedEfficiency, bonus, idleMinutes  

---

## 💡 Key Insights
- Planned efficiency & performance bonus are major productivity drivers.  
- Idle time negatively impacts output.  
- Balanced overtime enhances short-term results.  

---

## 🔮 Future Enhancements
- Apply **XGBoost / LightGBM**  
- **Hyperparameter tuning**  


---

## 🧰 Tech Stack
Python | pandas | numpy | matplotlib | seaborn | scikit-learn | Colab 

---

## 🏁 Conclusion
A complete ML pipeline predicting **factory team efficiency**, providing actionable insights for better workforce and production management.

---

# 🧠 Task 2: Retail Web Session Intelligence (RWSI) — Conversion Prediction  

---

## 🚀 Overview
This project analyzes customer web sessions to **predict purchase conversions** on a retail platform.  
It leverages behavioral and contextual session data to support data-driven marketing strategies.

---

## 🧩 Dataset Summary
- **File:** `rwsi_data(in).csv`
- **Goal:** Predict `MonetaryConversion` (Yes/No)
- **Records:** Thousands of web sessions
- **Features:**  
  - 🧮 **Numerical:** 'AdClicks', 'InfoSectionCount', 'InfoSectionTime','HelpPageVisits', 'HelpPageTime', 'ItemBrowseCount',
                      'ExitRateFirstPage','SessionExitRatio','PageEngagementScore','HolidayProximityIndex',TrafficSourceCode'	
  - 🏷️ **Categorical:** 'SessionID','VisitMonth','UserPlatformID','WebClientCode','MarketZone','UserCategory','MonetaryConversion'

---

## 🔍 Exploratory Data Analysis (EDA)

### Univariate Analysis
- Right-skewed distributions (`ItemBrowseTime`, `EngagementScore`).
- Outliers addressed later using IQR.
- **Categorical Highlights:**
  - `VisitMonth`: Peaks in **March, May, November, December**
  - `UserPlatformID`: Dominated by **Android & iOS**
  - `MarketZone`: Highest conversions in **North America, Asia-Pacific**

### Bivariate Analysis
- Higher conversion likelihood among:
  - **Returning mobile users**
  - **Festive months**
  - **North America** region
- Engagement metrics correlate strongly with conversions.

---

## 🧹 Data Preprocessing
- **Missing Values:** Median / Mode imputation  
- **Outlier Treatment:** IQR method  
- **Skewness Fix:** PowerTransformer (Yeo-Johnson)  
- **Encoding:** Label encoding for target; One-hot for features  
- **Scaling:** MinMaxScaler  
- **Data Split:** 80% Train | 20% Test  

---

## ⚙️ Model Building
Classification models implemented to predict conversion outcomes.

| Model | Description |
|--------|-------------|
| Logistic Regression | Interpretable baseline |
| Decision Tree | Handles nonlinearity |
| Random Forest | Ensemble method for accuracy & robustness |

---

## 📊 Evaluation Metrics
| Metric | Description |
|--------|-------------|
| Accuracy | Correct predictions |
| Precision | Positive prediction accuracy |
| Recall | Sensitivity to conversions |
| F1-Score | Balance between precision and recall |
| ROC-AUC | Class separation measure |

---

## 🧠 Model Comparison
| Model | Accuracy | ROC-AUC | Observation |
|--------|-----------|----------|--------------|
| Logistic Regression | ~0.86 | ~0.85 | Good baseline |
| Decision Tree | ~0.86 | ~0.87 | Nonlinear capture |
| Random Forest | **~0.88** | **~0.90** | Best performer |

✅ **Random Forest Classifier** delivered the highest accuracy and generalization.

---

## 📈 Visualization Highlights
- **Correlation Heatmap:** Feature interdependence  
- **Bar Plots:** Conversion trends per category  
- **ROC Curve:** Model comparison  
- **Feature Importance:** engagementScore, ItemBrowseTime, UserPlatformID  

---


## 🔮 Future Enhancements
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- Try **Gradient Boosting (XGBoost / LightGBM)**  

---

## 🧰 Tech Stack
Python | pandas | numpy | scikit-learn | matplotlib | seaborn | colab

---

## 🏁 Conclusion
An end-to-end ML system for predicting **customer conversion likelihood**, enabling better targeting and business optimization.

---

# ⚙️ Combined Overview

| Task | Type | Model | Best Accuracy / R² | Key Insight |
|------|------|--------|--------------------|--------------|
| 🏭 Manufacturing | Regression | Random Forest | R² ≈ 0.77 | Efficiency depends on planned output|
| 🧠 Retail Web Session | Classification | Random Forest | Accuracy ≈ 0.88 | Conversions depend on engagement |

---

## ⭐ How to Use
1. Clone the repository  
   ```bash
   git clone https://github.com/<your-username>/MACHINE-LEARNING-TASK.git

2. Navigate to the desired task

   ```bash
   cd MACHINE-LEARNING-TASK/task1   # or task2
   ```
3. Open the Jupyter Notebook

   ```bash
   jupyter notebook manufacturing.ipynb
   ```
4. Run all cells to reproduce results.

---

## 🏆 Final Note

Both projects demonstrate complete **Machine Learning lifecycles** —
from **raw data to model interpretation**, producing reliable predictions and insights
for **real-world decision-making** across **industrial** and **retail** domains.
