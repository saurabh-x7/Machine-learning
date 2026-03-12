# ================================
# House Price Prediction
# Simple Linear Regression
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_absolute_error, mean_squared_error,r2_score

# -------------------------------
# 1. LOADING THE DATASET
# -------------------------------

data = pd.read_csv("House_Price_Data.csv")

print("\nDataset Preview:")
print(data.head())

# -------------------------------
# 2. SELECTING FEATURE & TARGET
# -------------------------------


X = data[['Square_Feet']]      # independent variable
y = data['Price']       # dependent variable

# -------------------------------
# 3. TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. TRAINING THE MODEL
# -------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 5. PREDICTIONS
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# 6. MODEL ACCURACY
# -------------------------------

r2 = r2_score(y_test, y_pred)
print(f"\nModel R2 Score: {r2:.4f}")
print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

# -------------------------------
# 7. VISUALIZATION
# -------------------------------

plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', label="Actual Price")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Regression Line")
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression - House Price")
plt.legend()
plt.grid()

plt.show()

print("exiting....")