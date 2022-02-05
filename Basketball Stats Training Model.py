#https://www.youtube.com/watch?v=7eh4d6sabA0

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('basketball_win_loss.csv')
X = data.drop(columns=['W/L'])
Y = data['W/L']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
score = accuracy_score(Y_test, predictions)
print(score)