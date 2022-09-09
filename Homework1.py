import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
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
plt1.write_html(file="scatter.html", include_plotlyjs="cdn")
plt1.show()

plt2 = px.violin(data, x="Petal Length", y="Petal Width", color="Class")
plt2.write_html(file="violin.html", include_plotlyjs="cdn")
plt2.show()

plt3 = px.density_heatmap(data, x="Class", y="Petal Length")
plt3.write_html(file="heatmap.html", include_plotlyjs="cdn")
plt3.show()

plt4 = px.box(data, x="Sepal Length", y="Petal Length", color="Class")
plt4.write_html(file="box.html", include_plotlyjs="cdn")
plt4.show()

plt5 = px.bar(data, x="Petal Length", y="Petal Width", color="Class")
plt5.write_html(file="bar.html", include_plotlyjs="cdn")
plt5.show()


# Random Forest, Naive Bayes, SVM Modeling
X = data[["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]]
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


pipe1 = Pipeline(
    [("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100))]
)
pipe1.fit(X_train, y_train)
Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100)),
    ]
)
print("RandomForest/Pipeline Score: ", pipe1.score(X_test, y_test))


pipe2 = Pipeline([("scaler", StandardScaler()), ("gnb", GaussianNB())])
pipe2.fit(X_train, y_train)
Pipeline(steps=[("scaler", StandardScaler()), ("gnb", GaussianNB())])
print("Naive Bayes/Pipeline Score: ", pipe2.score(X_test, y_test))


pipe3 = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
pipe3.fit(X_train, y_train)
Pipeline(steps=[("scaler", StandardScaler()), ("svc", SVC())])
print("SVM/Pipeline Score: ", pipe3.score(X_test, y_test))
