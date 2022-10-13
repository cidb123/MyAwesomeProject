import pandas as pd
import plotly.graph_objects as go
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression


def dftest():
    # imported random dataset
    test = pd.read_csv("test.csv")
    test["HeartDisease"] = test.HeartDisease.astype("category")
    test["HeartDisease"] = test["HeartDisease"].map({"Yes": 1, "No": 0})
    test["Sex"] = test["Sex"].map({"Male": 1, "Female": 0})
    test["AlcoholDrinking"] = test["AlcoholDrinking"].map({"Yes": 1, "No": 0})

    response_list = [test["HeartDisease"]]
    predictor_list = [test["AlcoholDrinking"], test["Sex"], test["BMI"]]

    # testing
    resp_cont_list = []
    resp_cat_list = []
    for i in response_list:
        if i.dtype.name == "category":
            print(i.name, " is categorical")
            v1 = i
            resp_cat_list.append(i)
        else:
            print(i.name, " is continuous")
            resp_cont_list.append(i)
    pred_cont_list = []
    pred_bool_list = []
    for i in predictor_list:
        if len(pd.unique(i)) < 3:
            print(i.name, " is bool")
            pred_bool_list.append(i)
        else:
            print(i.name, " is continuous")
            pred_cont_list.append(i)
            v2 = i
            res = ttest_ind(v1, v2)
            print(i.name, "T test and P value", res)
            X = v2.values.reshape(-1, 1)
            y = v1
            clf = LogisticRegression(random_state=1).fit(X, y)
            logreg = clf.predict(X)
            badscore = clf.score(X, y)
            print(logreg)
            print(badscore)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=i, y=v2, mode="markers", name="lines"))
            fig.add_trace(
                go.Scatter(x=logreg, y=v2, mode="lines+markers", name="lines")
            )
            fig.show()
    return


dftest()
