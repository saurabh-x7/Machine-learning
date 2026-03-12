# =========================================
# House Price Prediction
# Multiple Linear Regression
# =========================================

import pandas as pd
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
# 2. SELECTING FEATURES & TARGET
# -------------------------------

X = data[[
    'Square_Feet',
    'Bedrooms',
    'Bathrooms',
    'Year_Built',
    'Garage_Size',
    'Location_Score'
]]

y = data['Price']

# -------------------------------
# 3. TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. TRAINING MODEL
# -------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 5. PREDICTIONS
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# 6. ACCURACY
# -------------------------------

r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print(f"\nModel R2 Score: {r2:.4f}")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

# -------------------------------
# 7. VISUALIZATION (Predicted vs Actual)
# -------------------------------

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression - Actual vs Predicted")
plt.grid()

plt.show()
