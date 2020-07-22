from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd


def load_csv(file_name):
    return pd.read_csv(file_name, header = 0, encoding='ISO-8859-1')
df = load_csv('train.csv')

feature_names = df.columns.tolist()
X = df.drop('Difficulty', 1)
Y = df['Difficulty']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor = RandomForestClassifier(n_estimators=5000, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
