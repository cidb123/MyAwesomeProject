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
from sklearn.model_selection import train_test_split
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
    SELECT   *
    FROM
        (SELECT a.game_id, a.team_id, a.opponent_team_id,tr.local_date,tr.win_lose, SUM(a.Hit)/SUM(a.atBat) AS Home_Average,
             a.atBat/a.Home_Run as Home_abperHR, SUM(a.Walk + a.Hit +a.Hit_By_Pitch)/a.atBat as Home_OBP,
             a.plateApperance/a.Strikeout as Home_abperK,
             SUM(a.Ground_Out+ a.Groundout + a.Grounded_Into_DP)/SUM(a.Line_Out + a.Line_Out + a.Fly_Out +a.Flyout + a.Pop_Out + a.Sac_Fly)
             as Home_GOperAO
        FROM team_batting_counts a
        Join team_results tr on a.team_id = tr.team_id and a.game_id = tr.game_id
        WHERE homeTeam = 1
        group by a.game_id, a.team_id
        ) a
    JOIN (SELECT b.game_id,SUM(b.Hit)/SUM(b.atBat) AS away_av,
             b.atBat/b.Home_Run as Away_abperHR, SUM(b.Walk + b.Hit +b.Hit_By_Pitch)/b.atBat as Away_OBP,
             b.plateApperance/Nullif(b.Strikeout,0) as Away_abperK,
             SUM(b.Ground_Out+ b.Groundout + b.Grounded_Into_DP)/Nullif(SUM(b.Line_Out + b.Line_Out + b.Fly_Out + b.Flyout + b.Pop_Out + b.Sac_Fly),0)
             as Away_GOperAO
        FROM team_batting_counts b
        Join team_results tr on b.team_id = tr.team_id and b.game_id = tr.game_id
        Where homeTeam = 0
        group by b.game_id, b.team_id
        ) b
    JOIN (SELECT   c.game_id, c.Strikeout/9 AS Home_kperinning,
             c.Strikeout/c.Walk as Home_kperbb, SUM(c.Walk + c.Hit +c.Hit_By_Pitch)/9 as Home_whip,
             c.Hit/c.atBat as Home_oba,
             c.Home_Run/9
             as Home_hrper9inning
        FROM team_pitching_counts c
        Join team_results tr on c.team_id = tr.team_id and c.game_id = tr.game_id
        WHERE c.homeTeam = 1
        GROUP BY c.game_id, c.team_id) c
    JOIN (SELECT   d.game_id, d.Strikeout/9 AS away_kperinning,
             d.Strikeout/d.Walk as away_kperbb, SUM(d.Walk + d.Hit +d.Hit_By_Pitch)/9 as away_whip,
             d.Hit/d.atBat as away_oba,
             d.Home_Run/9
             as away_hrperinning
        FROM team_pitching_counts d
        Join team_results tr on d.team_id = tr.team_id and d.game_id = tr.game_id
        WHERE d.homeTeam = 0
        GROUP BY d.game_id, d.team_id) d
    
        on b.game_id = a.game_id and b.game_id= c.game_id and b.game_id = d.game_id   
        """
    )
    cols = [
        "a.game_id",
        "team_id",
        "opponent_id",
        "local_date",
        "win",
        "Home_ave",
        "Home_ABperHr",
        "Home_OBP",
        "Home_ABperK",
        "Home_GOperAp",
        "b.game_id",
        "Away_ave",
        "Away_ABperHR",
        "Away_OBP",
        "Away_ABperK",
        "Away_GOperAO",
        "c.game_id",
        "Home_Kperinning",
        "Home_KperBB",
        "Home_WHIP",
        "Home_OBA",
        "Home_HRper9",
        "d.game_id",
        "Away_Kperinning",
        "Away_KperBB",
        "Away_WHIP",
        "Away_OBA",
        "Away_HRper9",
    ]
    ###########################
    # UPDATING DATAFRAME
    ###########################
    df = pd.DataFrame(cur.fetchall(), columns=cols)

    df = df.rename({"a.game_id": "game_id"}, axis=1)
    df = df.drop(["b.game_id", "c.game_id", "d.game_id"], axis=1)
    df = df.replace({"win": {"W": 1, "L": 0}})
    df["win"] = df["win"].astype(int)
    df = df.fillna(0)
    df["local_date"] = pd.to_datetime(df["local_date"])
    df = df.sort_values(by=["local_date"], ascending=[True])
    idx_df = df.drop(["game_id", "team_id", "opponent_id", "local_date"], axis=1)
    for col in idx_df.columns:
        idx_df[col] = pd.to_numeric(idx_df[col], errors="coerce")

    response = ["win"]
    cat_col = []
    cont_col = []
    cat_response = []
    cont_response = []

    for i in idx_df:
        if len(idx_df[i].unique()) < 5:
            cat_col.append(i)
        elif len(idx_df[i].unique()) > 5:
            cont_col.append(i)

    for z in response:
        if z in cat_col:
            cat_response.append(z)
            cat_col.remove(z)
        elif z in cont_col:
            cont_response.append(z)
            cont_col.remove(z)
    ###########################
    # CORRELATION
    ###########################
    cont_cont_pair = list(itertools.combinations(cont_col, 2))
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
    for item in cont_cont_pair:
        item = list(item)
        item.insert(0, "win")
        cont_cont_tuples.append(tuple(item))

    for items1 in cont_cont_tuples:
        resp_oo_df[items1] = pd.DataFrame()
        resp_oo_df[items1] = idx_df.loc[:, items1]

    cot_mse_dict = {}

    for key, value in resp_oo_df.items():
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
        dfm = resp_oo_df[key].iloc[bin_diff]
        dfm["pred. 1"] = bins_A[(tagsA - 1)[bin_diff]]
        dfm["pred. 2"] = bins_B[(tagsB - 1)[bin_diff]]
        df_out = dfm.groupby(["pred. 1", "pred. 2"])["win"].describe()
        cot_mse_dict[key] = df_out.iloc[:, [0, 1]]

    pop_mean = df.win.sum() / len(df.win)

    for key, value in cot_mse_dict.items():
        value["Bin Diff Mean"] = value.apply(
            lambda row: (pop_mean - row["mean"]) ** 2, axis=1
        )

    for key, value in cot_mse_dict.items():
        value["Diff Mean Resp Pair"] = value["Bin Diff Mean"].sum() / 5
    ###########################
    # LOGISTIC REGRESSION FOR EACH PAIR
    ###########################
    pt_df = pd.DataFrame(index=cont_col, columns=cont_col)

    for val1, val2 in cont_cont_pair:
        X = idx_df[[val1, val2]]
        Y = idx_df[response]
        X = sm.add_constant(X)

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
    X = pred_df.values
    y = idx_df[response].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

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
    print("Random Forest Accuracy: ", precision_score(y_test, y_pred))
    print("Random Forest Accuracy: ", recall_score(y_test, y_pred))

    clfsvm = svm.SVC(kernel="rbf")
    clfsvm.fit(X_train, y_train)
    y_predsvm = clfsvm.predict(X_test)
    print("SVM Accuracy:", metrics.accuracy_score(y_test, y_predsvm))
    print("Precision:", metrics.precision_score(y_test, y_predsvm))
    print("Recall:", metrics.recall_score(y_test, y_predsvm))

    """ The Support Vector machine preforms slightly better that the Random forest, and this was tested using other SVM
    Kernels like sigmoid and linear. I also tried using different amounts of trees, but none were as accurate and precise
    as the SVM.
    """

    brute_ls = []

    for key, value in cot_mse_dict.items():
        brute = value["Diff Mean Resp Pair"].iloc[0]
        brute_ls.append(brute)

    brute_df = oo_df2
    brute_df["Diff Mean Resp"] = brute_ls
    brute_df = brute_df.drop(["Correlation"], axis=1)

    dtree = pd.DataFrame()
    dtree["Model"] = ["Random Forest", "Support Vector"]
    dtree["Accuracy"] = [
        accuracy_score(y_test, y_pred),
        metrics.accuracy_score(y_test, y_predsvm),
    ]
    dtree["Precision"] = [
        precision_score(y_test, y_pred),
        metrics.precision_score(y_test, y_predsvm),
    ]
    dtree["Recall"] = [
        recall_score(y_test, y_pred),
        metrics.recall_score(y_test, y_predsvm),
    ]
    ###########################
    # HTML
    ###########################
    pio.write_html(fig2, file="fig2.html", auto_open=False)

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
          <iframe id="igraph" scrolling="yes" style="border:none;" seamless="seamless" 
          src="file:///Users/cidbrownell/src/MyAwesomeProject/fig2.html" height="525" width="100%">
          </iframe> 
          </body> 
    
    
    
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
    text_file = open("index.html", "w")
    text_file.write(html.format(corr_table=oo_df2.to_html(classes="css", index=False)))
    text_file.write(
        html2.format(brute_table=pt_df2.to_html(classes="css", index=False))
    )
    text_file.write(
        html3.format(dtree_results=dtree.to_html(classes="css", index=False))
    )

    text_file.close()

    new = 2
    url = "file:///Users/cidbrownell/src/MyAwesomeProject/index.html"
    webbrowser.open(url, new=new)


if __name__ == "__main__":
    sys.exit(main())
