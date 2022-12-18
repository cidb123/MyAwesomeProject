#!/bin/env python3
import itertools
import sys
import webbrowser

import mariadb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import statsmodels.api as sm
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None

pio.renderers.default = "browser"


def main():
    ############################
    # Connection
    ############################
    try:
        conn = mariadb.connect(
            user="root",
            password="root",  # pragma: allowlist secret
            host="localhost",
            port=3306,
            database="baseball",
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)

        # Get Cursor
    cur = conn.cursor()
    cur.execute(
        """ 
    
    SELECT * FROM features;
    
    
    """
    )
    cols = [
        "team_id",
        "win",
        "stadium_id",
        "game_id",
        "local_date",
        "home_batting_avg",
        "home_hrperab",
        "home_kperab",
        "home_OPS",
        "home_BACON",
        "away_batting_avg",
        "away_hrperab",
        "away_kperab",
        "away_OPS",
        "away_BACON",
        "home_hits9",
        "home_HR9",
        "home_Pop9",
        "home_POtoFO",
        "home_whip",
        "home_KtoW",
        "away_hits9",
        "away_HR9",
        "away_Pop9",
        "away_POtoFO",
        "away_whip",
        "away_KtoW",
        "adj_home_batting_avg",
        "adj_home_hrperab",
        "adj_home_OPS",
        "adj_home_BACON",
        "adj_away_batting_avg",
        "adj_away_hrperab",
        "adj_away_OPS",
        "adj_away_BACON",
        "adj_home_whip",
        "adj_away_whip",
    ]
    ###########################
    # UPDATING DATAFRAME
    ###########################

    df = pd.DataFrame(cur.fetchall(), columns=cols)

    df = df.dropna()
    df["local_date"] = pd.to_datetime(df["local_date"])
    df = df.sort_values(by=["local_date"], ascending=[True])

    df_basic = df.iloc[:, 0:15]
    df_adj = df.iloc[:, 0:26]
    idx_df = df_basic.drop(["game_id", "team_id", "stadium_id", "local_date"], axis=1)
    adj_idx_df = df_adj.drop(
        [
            "game_id",
            "team_id",
            "stadium_id",
            "local_date",
            "home_batting_avg",
            "home_hrperab",
            "home_OPS",
            "home_BACON",
            "away_batting_avg",
            "away_hrperab",
            "away_OPS",
            "away_BACON",
            "home_whip",
            "away_whip",
        ],
        axis=1,
    )

    for col in idx_df.columns:
        idx_df[col] = pd.to_numeric(idx_df[col], errors="coerce")
    for col in adj_idx_df.columns:
        adj_idx_df[col] = pd.to_numeric(adj_idx_df[col], errors="coerce")

    response = ["win"]
    cat_col = []
    cont_col = []
    cat_response = []
    cont_response = []
    response2 = ["win"]
    cat_col2 = []
    cont_col2 = []
    cat_response2 = []
    cont_response2 = []

    for i in idx_df:
        if len(idx_df[i].unique()) < 5:
            cat_col.append(i)
        elif len(idx_df[i].unique()) > 5:
            cont_col.append(i)

    for i in adj_idx_df:
        if len(adj_idx_df[i].unique()) < 5:
            cat_col2.append(i)
        elif len(adj_idx_df[i].unique()) > 5:
            cont_col2.append(i)

    for z in response:
        if z in cat_col:
            cat_response.append(z)
            cat_col.remove(z)
        elif z in cont_col:
            cont_response.append(z)
            cont_col.remove(z)
    for z in response2:
        if z in cat_col2:
            cat_response2.append(z)
            cat_col2.remove(z)
        elif z in cont_col2:
            cont_response2.append(z)
            cont_col2.remove(z)
    cont_cont_pair = list(itertools.combinations(cont_col, 2))
    cont_col2 = []
    for el in cont_col:
        sub = el.split(", ")
        cont_col2.append(sub)

    oo_df = pd.DataFrame(index=cont_col, columns=cont_col)
    for c, d in cont_cont_pair:
        corr_cont = idx_df.loc[:, c].corr(idx_df.loc[:, d], method="pearson")
        oo_df.loc[c, d] = corr_cont

    oo_df2 = oo_df.stack().reset_index()
    oo_df2.columns = ["Predictor 1", "Predictor 2", "Correlation"]
    oo_df2 = oo_df2.sort_values(by=["Correlation"], ascending=[False])

    oo_df = oo_df.T
    fig2 = px.imshow(
        oo_df, labels=dict(x="Continuous Continuous Correlation", color="Correlation")
    )
    fig2.update_xaxes(side="top")

    resp_oo_df = {}
    cont_cont_tuples = []
    ###########################
    # BRUTE FORCE
    ###########################
    for item in cont_col2:
        item.insert(0, "win")
        cont_cont_tuples.append(tuple(item))

    for items1 in cont_cont_tuples:
        resp_oo_df[items1] = pd.DataFrame()
        resp_oo_df[items1] = idx_df.loc[:, items1]

    cot_mse_dict = {}

    for key, value in resp_oo_df.items():
        bins_A = np.arange(
            value.iloc[:, 1].min(), value.iloc[:, 1].max(), value.iloc[:, 1].max() / 10
        )

        tagsA = np.searchsorted(bins_A, value.iloc[:, 1])
        dfm = resp_oo_df[key].iloc[tagsA]
        dfm["pred"] = bins_A[(tagsA - 1)]
        df_out = dfm.groupby(["pred"])["win"].describe()
        cot_mse_dict[key] = df_out.iloc[:, [0, 1]]

    pop_mean = df.win.sum() / len(df.win)

    for key, value in cot_mse_dict.items():
        value["Meansquareddiff"] = value.apply(
            lambda row: ((row["mean"] - pop_mean) ** 2), axis=1
        )

    for key, value in cot_mse_dict.items():
        value["popmean"] = pop_mean
        value["maybegraph"] = value.apply(lambda row: (row["mean"] - pop_mean), axis=1)
        value["Diff Mean Resp Pair"] = value["maybegraph"].sum() / 10
    ###########################
    # LOGISTIC REGRESSION FOR EACH PAIR
    ###########################
    pt_df = pd.DataFrame(index=cont_col, columns=cont_col)
    loglist = []
    for val1, val2 in cont_cont_pair:
        X = idx_df[[val1, val2]]
        Y = idx_df[response]

        X = sm.add_constant(X)
        model = sm.Logit(Y, X)
        results = model.fit(disp=False)
        pt_df.loc[val1, val2] = results.pvalues[1], results.tvalues[1]

    pt_df2 = pt_df.stack().reset_index()
    pt_df2.columns = ["Predictor 1", "Predictor 2", "P_values & T_score"]

    pt_df2 = pt_df2.sort_values(by=["P_values & T_score"], ascending=[True])

    ###########################
    # PREDICTION MODELS
    ###########################
    pred_df = idx_df.drop(response, axis=1)
    pred_df2 = adj_idx_df.drop(response, axis=1)

    X_train = pred_df.iloc[:9097].values
    X_test = pred_df.iloc[-3033:].values
    y_train = idx_df[response][:9097].values.ravel()
    y_test = idx_df[response][-3033:].values.ravel()

    X_train2 = pred_df2.iloc[:9097].values
    X_test2 = pred_df2.iloc[-3033:].values
    y_train2 = adj_idx_df[response2][:9097].values.ravel()
    y_test2 = adj_idx_df[response2][-3033:].values.ravel()

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Random Forest Accuracy: ", accuracy_score(y_test, y_pred))
    print("Random Forest Precision: ", precision_score(y_test, y_pred))
    print("Random Forest Recall: ", recall_score(y_test, y_pred))

    clfsvm = svm.SVC(kernel="rbf")
    clfsvm.fit(X_train, y_train)
    y_predsvm = clfsvm.predict(X_test)
    print("SVM Accuracy:", metrics.accuracy_score(y_test, y_predsvm))
    print("Precision:", metrics.precision_score(y_test, y_predsvm))
    print("Recall:", metrics.recall_score(y_test, y_predsvm))

    from sklearn.naive_bayes import GaussianNB

    clg = GaussianNB()
    clg.fit(X_train, y_train)
    y_predgaus = clg.predict(X_test)
    from sklearn.linear_model import LogisticRegression

    clx = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_predclx = clx.predict(X_test)

    """ The Support Vector machine preforms slightly better that the Random forest, and this was tested using other SVM
    Kernels like sigmoid and linear. I also tried using different amounts of trees, but none were as accurate and precise
    as the SVM.
    """
    X_train2 = sc.fit_transform(X_train2)
    X_test2 = sc.transform(X_test2)

    regressor2 = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor2.fit(X_train2, y_train2)
    y_pred2 = regressor.predict(X_test)
    y_pred2 = np.where(y_pred2 > 0.5, 1, 0)

    clfsvm2 = svm.SVC(kernel="rbf")
    clfsvm2.fit(X_train2, y_train2)
    y_predsvm2 = clfsvm2.predict(X_test2)

    from sklearn.naive_bayes import GaussianNB

    clg2 = GaussianNB()
    clg2.fit(X_train2, y_train2)
    y_predgaus2 = clg2.predict(X_test2)
    from sklearn.linear_model import LogisticRegression

    clx2 = LogisticRegression(random_state=0).fit(X_train2, y_train2)
    y_predclx2 = clx2.predict(X_test2)

    """END OF NEW MODELS"""
    resp_oo_df2 = {}
    cont_cont_tuples2 = []
    for item in cont_cont_pair:
        item = list(item)
        item.insert(0, "win")
        cont_cont_tuples2.append(tuple(item))

    for items1 in cont_cont_tuples2:
        resp_oo_df2[items1] = pd.DataFrame()
        resp_oo_df2[items1] = idx_df.loc[:, items1]

    cot_mse_dict2 = {}

    for key, value in resp_oo_df2.items():
        bins_A = np.arange(
            value.iloc[:, 1].min(), value.iloc[:, 1].max(), value.iloc[:, 1].max() / 5
        )
        bins_B = np.arange(
            value.iloc[:, 2].min(), value.iloc[:, 2].max(), value.iloc[:, 2].max() / 5
        )
        tagsA = np.searchsorted(bins_A, value.iloc[:, 1])
        tagsB = np.searchsorted(bins_B, value.iloc[:, 2])
        bin_diff = (
            (tagsB > 0) & (tagsB < len(bins_B)) & (tagsA > 0) & (tagsA < len(bins_A))
        )
        dfm = resp_oo_df2[key].iloc[bin_diff]
        dfm["pred. 1"] = bins_A[(tagsA - 1)[bin_diff]]
        dfm["pred. 2"] = bins_B[(tagsB - 1)[bin_diff]]
        df_out = dfm.groupby(["pred. 1", "pred. 2"])["win"].describe()
        cot_mse_dict2[key] = df_out.iloc[:, [0, 1]]

    pop_mean = df.win.sum() / len(df.win)

    for key, value in cot_mse_dict2.items():
        value["Bin Diff Mean"] = value.apply(
            lambda row: (pop_mean - row["mean"]) ** 2, axis=1
        )

    for key, value in cot_mse_dict2.items():
        value["Diff Mean Resp Pair"] = value["Bin Diff Mean"].sum() / 5
    brute_ls = []

    for key, value in cot_mse_dict2.items():
        brute = value["Diff Mean Resp Pair"].iloc[0]
        brute_ls.append(brute)

    brute_df = oo_df2
    brute_df["Diff Mean Resp"] = brute_ls
    brute_df = brute_df.drop(["Correlation"], axis=1)

    dtree = pd.DataFrame()
    dtree["Model"] = [
        "Random Forest",
        "Support Vector",
        "Naive Bayes",
        "Logistic Regression",
    ]
    dtree["Accuracy"] = [
        accuracy_score(y_test, y_pred),
        metrics.accuracy_score(y_test, y_predsvm),
        metrics.accuracy_score(y_test, y_predgaus),
        metrics.accuracy_score(y_test, y_predclx),
    ]
    dtree["Precision"] = [
        precision_score(y_test, y_pred),
        metrics.precision_score(y_test, y_predsvm),
        metrics.precision_score(y_test, y_predgaus),
        metrics.precision_score(y_test, y_predclx),
    ]
    dtree["Recall"] = [
        recall_score(y_test, y_pred),
        metrics.recall_score(y_test, y_predsvm),
        metrics.recall_score(y_test, y_predgaus),
        metrics.recall_score(y_test, y_predclx),
    ]
    dtree2 = pd.DataFrame()
    dtree2["Model with Park Factor"] = [
        "Random Forest",
        "Support Vector",
        "Naive Bayes",
        "Logistic Regression",
    ]
    dtree2["Accuracy"] = [
        accuracy_score(y_test2, y_pred2),
        metrics.accuracy_score(y_test2, y_predsvm2),
        metrics.accuracy_score(y_test2, y_predgaus2),
        metrics.accuracy_score(y_test2, y_predclx2),
    ]
    dtree2["Precision"] = [
        precision_score(y_test2, y_pred2),
        metrics.precision_score(y_test2, y_predsvm2),
        metrics.precision_score(y_test2, y_predgaus2),
        metrics.precision_score(y_test2, y_predclx2),
    ]
    dtree2["Recall"] = [
        recall_score(y_test2, y_pred2),
        metrics.recall_score(y_test2, y_predsvm2),
        metrics.recall_score(y_test2, y_predgaus2),
        metrics.recall_score(y_test2, y_predclx2),
    ]
    ###########################
    # HTML
    ###########################
    idx_df["win"] = idx_df["win"].astype("category")
    box_list = []
    for i in cont_col:
        fig5 = px.box(idx_df, x=idx_df["win"].values, y=idx_df[i].values)
        fig5.update_layout(xaxis_title="Groupings(Win)", yaxis_title=i)
        fig5.write_html(i + ".html")
        hyperlink_format = '<a href="{link}">{text}</a>'
        box_list.append(i)
        box_list.append(hyperlink_format.format(link=i + ".html", text="link"))
    it = iter(box_list)
    both = zip(it, it)

    boxdf = pd.DataFrame(both, columns=["<tr><th>Pred</th>", "<th>box</th></tr>"])
    boxdf.to_html(escape=False)

    pio.write_html(fig2, file="fig2.html", auto_open=False)

    binlist = []
    for key, value in cot_mse_dict.items():
        key = list(key)
        value.reset_index(inplace=True)
        reset = value.rename(columns={"index": "pred"})

        y = reset["popmean"]

        fig66 = px.line(x=reset["pred"], y=y, labels={"y": "mean"})
        fig66.add_scatter(x=reset["pred"], y=reset["maybegraph"], name="mean-pop")
        popnew = []
        for i in reset["count"].values:
            u = i / 12130
            popnew.append(u)
        fig66.add_bar(x=reset["pred"], y=popnew, name="population")
        fig66.update_layout(
            title=key[1],
            xaxis_title="Bin",
            yaxis_title="Response",
            legend_title="lines",
        )
        key[1] = key[1] + "bin" + ".html"
        fig66.write_html(key[1])
        hyperlink_format2 = '<a href="{link}">{text}</a>'
        binlist.append(key[1])
        binlist.append(hyperlink_format2.format(link=key[1], text="link"))
    binit = iter(binlist)
    bothbin = zip(binit, binit)
    bin2df = pd.DataFrame(bothbin, columns=["Pred", "Bindiff"])
    bin2df.to_html(escape=False)

    help = fig2.to_html()

    html = """
            <html>
              <head><title>Hw5 page</title>
              </head>
              <link rel="stylesheet" type="text/css" href="css.css"/>

              <body style = text-align: center;> 
              <h1 style="font-family:Times;text-align: center;">
              Hw 5 Baseball Analysis
              </h1>
              <h2 style="font-family:Times;">
              Pearson's Correlation and Brute Force
              </h2>
              {corr_table} 
              <h2 style="font-family:Times;text-align: center;">
              Correlation Matrix 
              </h2><br>
              
              
              </body> 



            """

    html8 = """<br>
                      <h2 style="font-family:Times;">
                      </h2>
                      {matrix}


                    """
    html6 = """<br>
                      <h2 style="font-family:Times;">
                      Box Plots 
                      </h2>
                      <table>
                      {box_table} 
                      </table>


                    """

    html4 = """<br>
              <h2 style="font-family:Times;">
              Mean Diff tables 
              </h2>
              {bin_table} 


            """
    html2 = """<br>
                  <h2 style="font-family:Times;">
                  Logistic Regression Results 
                  </h2>
                  {brute_table} 


                """

    html3 = """<br><br>
              <h2 style="font-family:Times;">
              Model Results
              </h2>
              {dtree_results} 
              <h3 style="font-family:Times;text-align: center;">
              The Support Vector machine preforms slightly better that the Random forest, with a .50 threshold. The results still 
              stay around 80% accurate when tested with a .75 threshold. This was tested using other SVM
              Kernels like sigmoid and linear. I also tried using different amounts of trees, but none were as accurate and precise
              as the SVM.
              </h3>
              <br>
              <br>
              <br>
              </body.
            </html>
            """

    html22 = """<br><br>
                  <h2 style="font-family:Times;">
                  Model Results With Park Factor
                  </h2>
                  {dtree_results2} 
                  """
    text_file = open("index.html", "w")
    text_file.write(html.format(corr_table=oo_df2.to_html(classes="css", index=False)))

    text_file.write(html8.format(matrix=help))
    text_file.write(html6.format(box_table=boxdf))

    text_file.write(html4.format(bin_table=bin2df))

    text_file.write(
        html2.format(brute_table=pt_df2.to_html(classes="css", index=False))
    )
    text_file.write(
        html3.format(dtree_results=dtree.to_html(classes="css", index=False))
    )
    text_file.write(
        html22.format(dtree_results2=dtree2.to_html(classes="css", index=False))
    )
    text_file.write(html4.format(bin_table=bin2df))

    text_file.close()

    import os

    webbrowser.open("file://" + os.path.realpath("index.html"))


if __name__ == "__main__":
    sys.exit(main())
