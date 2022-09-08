import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pio.renderers.default = "browser"

headers = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"]

data = pd.read_csv("iris.data", names=headers)

print("Mean Sepal Length: ", np.mean(data["Sepal Length"]))
print("Min Sepal Width: ", np.min(data["Sepal Width"]))
print("Max Petal Length: ", np.max(data["Petal Length"]))
print("0.25 Quantile Petal Width: ", np.quantile(data["Petal Width"], 0.25))

plt1 = px.scatter(data, x="Sepal Length", y="Sepal Width", color="Class")
plt1.show()

plt2 = px.violin(data, x="Petal Length", y="Petal Width", color="Class")
plt2.show()

plt3 = px.density_heatmap(data, x="Class", y="Petal Length")
plt3.show()

plt4 = px.box(data, x="Sepal Length", y="Petal Length", color="Class")
plt4.show()

plt5 = px.bar(data, x="Petal Length", y="Petal Width", color="Class")
plt5.show()


dummydata = pd.get_dummies(data)
scaler = StandardScaler()
print(scaler.fit(dummydata))
print(scaler.mean_)
print(scaler.transform(dummydata))
tfdata = scaler.transform(dummydata)

# Random Forest
X = data[["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]]
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy RF:", metrics.accuracy_score(y_test, y_pred))

# Naive Bayes

gnb = GaussianNB()
y_pred2 = gnb.fit(X_train, y_train).predict(X_test)
print(
    "Naive Bayes number of mislabeled points out of a total %d points : %d"
    % (X_test.shape[0], (y_test != y_pred2).sum())
)

# SVM/Pipeline

pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
pipe.fit(X_train, y_train)
Pipeline(steps=[("scaler", StandardScaler()), ("svc", SVC())])
print("SVM/Pipeline Score: ", pipe.score(X_test, y_test))
