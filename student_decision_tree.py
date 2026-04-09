#Importing libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Loading dataset and checking the dataset

df = pd.read_csv("student.csv", sep=';')
print(df.head())
print(df.shape())

df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# defining features and target

X = df[['studytime', 'failures', 'absences', 'G1', 'G2']]
y = df['pass']

#Split the dataset 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Createing the model and training the model

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

#Predictions

y_pred = model.predict(X_test)

#Evaluation/accuracy 

print("Accuracy:", accuracy_score(y_test, y_pred))

