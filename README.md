# 🛒 Walmart Weekly Sales Forecasting Using Machine Learning 📈

This project predicts **weekly sales** for Walmart stores using historical data and machine learning regression models. The goal is to support business decisions related to **inventory planning**, **holiday promotions**, and **store-level performance** optimization.

---

## 📂 Dataset

- **Source**: [Kaggle - Walmart Dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset)
- **File Used**: `Sales Forecasting.csv`

### 📋 Description
The dataset includes:
- Sales data across **45 Walmart stores** from **2010 to 2012**
- Features like:
  - `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`
  - `Holiday_Flag`, `Store`, `Weekly_Sales`
- **Target Variable**: `Weekly_Sales` (Regression Problem)

---

## ⚙️ Technologies Used

- Python (Jupyter Notebook)
- Libraries:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `joblib`

---

## 🔍 Project Workflow

### 🧾 1. Data Loading & Feature Extraction
- Parsed `Date` column into `Year`, `Month`, `Week`
- Dropped non-essential columns like `index`, `DayOfWeek`

### 🧹 2. Data Cleaning
- Verified data integrity
- No missing values
- Scaled features for Linear Regression using `StandardScaler`

---

## 📊 Exploratory Data Analysis (EDA)

### 📈 Distribution of Weekly Sales
- **Right-skewed** distribution
- Majority between **$400K–$1.2M**, with outliers > $3M

### 📦 Boxplot of Weekly Sales
- Median ≈ **$1.1M**
- Outliers show major peak weeks

### 🗓️ Weekly Sales Trends
- Stable trends across 2010–2012
- Spikes around **December and Week 47–50**
- January has the lowest average sales

### 🏬 Average Sales by Store
- **Stores 20, 4, and 2** are top performers
- Large variation in store-wise sales

### 🎉 Holiday Impact
- **Holiday weeks** show higher and more variable sales
- Not all holidays result in spikes

### 📅 Monthly Trends
- **December > June > February** in average sales
- **January** shows lowest sales

### 🔗 Correlation Heatmap
- Weak overall correlations
- `Store` had the strongest correlation at **-0.34**
- Indicates complex patterns best captured via models

---

## 🤖 Model Training & Evaluation

### 🧠 Features & Target
- **X** = All columns except `Weekly_Sales` and `index`
- **y** = `Weekly_Sales`

### 🔀 Train/Test Split
- Used `train_test_split(test_size=0.2, random_state=42)`

---

### 📉 Linear Regression (Baseline)

| Metric | Value |
|--------|--------|
| MAE    | \$431,771.48 |
| RMSE   | \$521,296.05 |
| R²     | 0.1565 |

> ❌ Baseline model, struggles with non-linear patterns.

---

### 🌲 Random Forest Regressor (Final Model)

| Metric | Value |
|--------|--------|
| MAE    | \$62,022.51 |
| RMSE   | \$114,106.34 |
| R²     | 0.9596 |

> ✅ Captures non-linear relationships. High accuracy and generalization.

---

### 📈 Actual vs Predicted Sales (Random Forest)
- Most predictions are near-perfect
- Red dashed line confirms ideal prediction alignment
- A few outliers may be linked to holidays or promotions

---

## 🔍 Feature Importance

Top features:
- `Store`, `CPI`, `Unemployment`

Less important:
- `Holiday_Flag`, `Month`, `Year`

> Dropping `index` improved performance and prevented leakage.

---

## 💾 Model Saving

Saved the final trained model using `joblib`:

```python
import joblib
joblib.dump(rf_model, 'random_forest_sales_model.pkl')
