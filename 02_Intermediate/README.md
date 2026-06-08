# Intermediate Task: Car Selling Price Prediction and Analysis

## 📌 Project Overview
This project is part of the **ShadowFox AIML Internship** (Intermediate Level). The goal is to build a high-performance machine learning regression model to accurately predict the approximate selling price of used cars based on multiple physical, operational, and market features. 

By implementing an automated end-to-end data pipeline, the model achieves a strong predictive fit, allowing for a precise evaluation of vehicular depreciation.

---

## 📊 Dataset Features
The model leverages the following attributes from the dataset to make its valuations:
* **Present_Price:** The original showroom price of the vehicle (Numerical).
* **Kms_Driven:** Total distance traveled by the car in kilometers (Numerical).
* **Fuel_Type:** Type of fuel used (Diesel / Petrol / CNG) (Categorical).
* **Seller_Type:** Type of seller (Dealer / Individual) (Categorical).
* **Transmission:** Gear transmission type (Manual / Automatic) (Categorical).
* **Owner:** Number of previous owners (Numerical/Ordinal).
* **Years_of_Service:** Engineered feature representing the total age/depreciation timeframe of the vehicle (Calculated from the manufacturing year).

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **IDE:** Spyder (Anaconda Ecosystem) / VS Code
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning framework:** `scikit-learn`

---

## 🚀 Machine Learning Pipeline Architecture
To ensure zero data leakage and elegant preprocessing, the script utilizes a robust `scikit-learn Pipeline`:
1.  **Feature Engineering:** Extracts asset age via `Years_of_Service = Current_Year - Manufacturing_Year` to model linear vehicle depreciation patterns.
2.  **Numerical Transformer:** Implements `StandardScaler` to normalize numeric variances in features like mileage and price.
3.  **Categorical Transformer:** Uses `OneHotEncoder(drop='first')` to cleanly map multi-class strings into boolean flags.
4.  **Model Selection:** An ensemble-based **Random Forest Regressor** (`n_estimators=100`) handles non-linear relationships and high-leverage target variances natively.

---

## 🏆 Performance Report & Results

Upon training on an 80/20 train-test split, the model achieved exceptional stability and fit metrics:

* **Mean Absolute Error (MAE):** 0.623 (Average valuation error variance)
* **Root Mean Squared Error (RMSE):** 0.941
* **R² Prediction Accuracy Score:** **0.9621 (96.21% Total Variance Captured)**

### Generated Insights & Artifacts
The pipeline automatically outputs and updates two visual plots saved directly in this folder:
1.  `feature_importance.png` – Quantifies exactly which parameters (such as `Present_Price` and vehicle age) weigh heaviest in the pricing algorithm.
2.  `actual_vs_predicted.png` – Provides a regression scatter plot displaying the near-perfect alignment between actual market values and model estimations.

---

## 📂 Repository Structure
```text
ShadowFox/
│
├── 01_Beginner_Task/
│   └── [Beginner files]
│
└── 02_Intermediate/
    ├── car_data.csv               # Raw Dataset
    ├── car_price_prediction.py    # Clean Production Script
    ├── README.md                  # Project Documentation
    ├── feature_importance.png     # Automated Plot 1
    └── actual_vs_predicted.png    # Automated Plot 2
