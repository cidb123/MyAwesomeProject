import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

pio.renderers.default = "browser"


def __main__():
    def dftest():
        # testing random dataset
        test = pd.read_csv("test.csv")
        test["HeartDisease"] = test.HeartDisease.astype("category")
        test["HeartDisease"] = test["HeartDisease"].map({"Yes": 1, "No": 0})
        test["Sex"] = test["Sex"].map({"Male": 1, "Female": 0})
        test["AlcoholDrinking"] = test["AlcoholDrinking"].map({"Yes": 1, "No": 0})

        response_list = [test["HeartDisease"]]
        predictor_list = [test["AlcoholDrinking"], test["Sex"], test["BMI"]]

        # categorizing
        resp_cont_list = []
        resp_cat_list = []
        for i in response_list:
            if len(pd.unique(i)) < 3:
                print(i.name, " is categorical")
                v1 = i
                resp_cat_list.append(i)
            else:
                print(i.name, " is continuous")
                v1 = 1
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
                if len(resp_cat_list) == 1:
                    clf = LogisticRegression(random_state=0).fit(X, y)
                    logreg = clf.predict(X)
                    log_score = clf.score(X, y)
                    print("logistic Regression: ", log_score)
                    points = go.Scatter(
                        name="Logistic Data",
                        x=X,
                        y=y,
                        mode="markers",
                        marker=dict(
                            symbol="line-ns",
                            size=5,
                            line=dict(width=1, color="darkblue"),
                        ),
                    )
                    layout = dict(
                        xaxis=dict(title="X values"), yaxis=dict(title="Logistic Y")
                    )
                    go.Figure(data=[points], layout=layout)
                    fig = go.Figure([points], layout=layout)
                    logline = go.Scatter(
                        name=" Logistic Regression",
                        x=X,
                        y=logreg,
                        line=dict(color="cyan", width=3),
                    )
                    fig.add_trace(logline)
                    fig.show()
                else:
                    reg = LinearRegression().fit(X, y)
                    reg_score = reg.score(X, y)
                    print("Regression Score: ", reg_score)
                    points2 = go.Scatter(
                        name="Linear Data",
                        x=X,
                        y=y,
                        mode="markers",
                        marker=dict(
                            symbol="line-ns",
                            size=5,
                            line=dict(width=1, color="darkblue"),
                        ),
                    )
                    layout2 = dict(xaxis=dict(title="X values"), yaxis=dict(title="Y"))
                    go.Figure(data=[points2], layout=layout2)
                    fig = go.Figure([points2], layout=layout2)
                    line = go.Scatter(
                        name="Regression", x=X, y=reg, line=dict(color="cyan", width=3)
                    )
                    fig.add_trace(line)
                    fig.show()
                    feature_names = [f"feature {i}" for i in range(X.shape[1])]
                    forest = RandomForestClassifier(random_state=0)
                    importance = forest.feature_importances_
                    std = np.std(
                        [tree.feature_importances_ for tree in forest.estimators_],
                        axis=0,
                    )
                    forest_importance = pd.Series(importance, index=feature_names)
                    forest.fit(X, y)

                    fig2 = px.bar(forest_importance, x=std, y=y)
                    fig2.show()

        return

    dftest()


if __name__ == "__main__":
    sys.exit(__main__())
