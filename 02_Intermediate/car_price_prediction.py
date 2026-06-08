"""
ShadowFox AIML Internship
Task Level: Intermediate (Task 2)
Project: Car Selling Price Prediction and Analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set visualization style
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("--- Step 1: Loading Data ---")
data_path = "car_data.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find '{data_path}'. Make sure it's in the correct folder!")

df = pd.read_csv(data_path)

print("\n--- Step 2: Feature Engineering ---")
current_year = 2026 

if 'Year' in df.columns:
    df['Years_of_Service'] = current_year - df['Year']
    df = df.drop(columns=['Year'])
    print("Engineered 'Years_of_Service' column successfully.")

if 'Car_Name' in df.columns:
    df = df.drop(columns=['Car_Name'])

print("\n--- Step 3: Splitting Data ---")
target_col = 'Selling_Price'
X = df.drop(columns=[target_col])
y = df[target_col]

num_features = ['Present_Price', 'Kms_Driven', 'Owner', 'Years_of_Service']
cat_features = ['Fuel_Type', 'Seller_Type', 'Transmission']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Step 4: Training Model ---")
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
print("Model training completed!")

print("\n--- Step 5: Evaluation ---")
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n==========================================")
print("             PERFORMANCE REPORT           ")
print("==========================================")
print(f"Mean Absolute Error (MAE):    {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Prediction Accuracy Score:   {r2:.4f} ({r2*100:.2f}%)")
print("==========================================")

print("\n--- Step 6: Saving Visualizations ---")
encoded_cat_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
all_feature_names = num_features + encoded_cat_features
importances = pipeline.named_steps['model'].feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=[all_feature_names[i] for i in indices], palette="viridis")
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="b")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Actual vs. Predicted Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("[Success] Execution finished! Images saved successfully.")