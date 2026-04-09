#Importing libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#Loading dataset from sklearn library and checking the dataset

data = load_breast_cancer()   
print(data.feature_names)       # checking features
df = pd.DataFrame(data.data, columns=data.feature_names)  
df['target'] = data.target
print(df.head())

# defining features and target

X = data.data        # Features, Features are already defined in the dataset
y = data.target      # Labels (0 = malignant, 1 = benign), same a Features the target is also defined

#Split the dataset 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Createing the model and Training model

model = SVC(kernel='linear')
model.fit(X_train, y_train)

#Predictions

y_pred = model.predict(X_test)

#Evaluation/accuracy 

print("Accuracy:", accuracy_score(y_test, y_pred))
