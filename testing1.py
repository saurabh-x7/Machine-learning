import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ----------------------
# 1. LOAD DATA
# ----------------------

df = pd.read_csv(r"C:\Users\Ayush\Desktop\projects\iot project\Air quality index prediction\dataset\cleaned_aqi_data.csv")

print("Original shape:", df.shape)


# ----------------------
# 2. REMOVE ROWS WHERE TARGET IS NaN
# ----------------------

print("After removing NaN targets:", df.shape)
print("\nClass distribution:")
print(df['aqi_bucket'].value_counts())

# ----------------------
# 3. PREPARE FEATURES
# ----------------------

# Drop non-numeric useless columns
df_model = df.drop(columns=['City', 'Date'], errors='ignore')

# Encode target labels
le = LabelEncoder()
df_model['aqi_bucket'] = le.fit_transform(df_model['aqi_bucket'])

# Split X and y
X = df_model.drop(columns=['AQI', 'aqi_bucket'], errors='ignore')
y = df_model['aqi_bucket']

# ----------------------
# 4. TRAIN TEST SPLIT
# ----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ----------------------
# 5. SCALING
# ----------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------
# 6. LOGISTIC REGRESSION
# ----------------------

model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# ----------------------
# 7. EVALUATION
# ----------------------

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------
# 8. CONFUSION MATRIX GRAPH
# ----------------------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.show()

# ----------------------
# 9. TESTING WITH NEW SAMPLE (CLEAN VERSION)
# ----------------------

sample_dict = {
    'PM2.5': 80,
    'PM10': 120,
    'NO': 20,
    'NO2': 40,
    'NOx': 60,
    'NH3': 25,
    'CO': 1.2,
    'SO2': 10,
    'O3': 50,
    'Benzene': 2,
    'Toluene': 5
}

sample_df = pd.DataFrame([sample_dict])

# Scale properly
sample_scaled = scaler.transform(sample_df)

# Predict
prediction = model.predict(sample_scaled)
predicted_label = le.inverse_transform(prediction)

print("\nPredicted AQI Bucket:", predicted_label[0])
