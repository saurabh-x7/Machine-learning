

# Step 1: Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset
df = pd.read_csv("accident_prediction_india.csv")

print("First 5 rows:")
print(df.head())

# 2) Check Missing Values
# ----------------------------
print("\nMissing values in each column:")
print(df.isnull().sum())

# Step 3: Create Binary Target
# Convert Accident Severity into numeric
# Minor = 0, Serious = 1

df['Accident Severity'] = df['Accident Severity'].map({
    'Minor': 0,
    'Serious': 1
})

# Remove rows where severity might be missing
df = df.dropna(subset=['Accident Severity'])



# 3) Set correct target column
target = "Accident Severity"
print("Target Column:", target)



# Step 4: Select important features
features = [
    'Speed Limit (km/h)',
    'Driver Age',
    'Number of Vehicles Involved',
    'Number of Casualties',
    'Number of Fatalities',
    'Alcohol Involvement'
]

df = df[features + ['Accident Severity']]

# Convert Alcohol Yes/No to 1/0
df['Alcohol Involvement'] = df['Alcohol Involvement'].map({
    'Yes': 1,
    'No': 0
})

# Remove any remaining missing values
df = df.dropna()

# Step 5: Define X and y
X = df.drop('Accident Severity', axis=1)
y = df['Accident Severity']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Step 9: Predictions
y_pred = model.predict(X_test_scaled)

# Step 10: Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 11: Sample Prediction
sample = np.array([[80, 30, 2, 1, 0, 1]])  
# Format:
# Speed, Age, Vehicles, Casualties, Fatalities, Alcohol

sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("\nPredicted Severity (1=Serious, 0=Minor):", prediction[0])

