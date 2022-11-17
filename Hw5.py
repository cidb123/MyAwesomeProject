import itertools
import sys

import mariadb
import pandas as pd
import plotly.express as px


def __main__():
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
        Select * from batter_total
    """
    )

    df = pd.DataFrame(cur.fetchall())
    print(df)

    response = ["win_lose"]
    cat_col = []
    cont_col = []
    cat_response = []
    cont_response = []

    # important loop that determines categorical variables.
    for i in df:
        if len(df[i].unique()) < 10:
            cat_col.append(i)
        elif len(df[i].unique()) > 10:
            cont_col.append(i)

    for z in response:
        if z in cat_col:
            cat_response.append(z)
            cat_col.remove(z)
        elif z in cont_col:
            cont_response.append(z)
            cont_col.remove(z)

    cont_cont_pair = list(itertools.combinations(cont_col, 2))

    oo_df = pd.DataFrame(index=cont_col, columns=cont_col)

    for c, d in cont_cont_pair:
        corr_cont = df.loc[:, c].corr(df.loc[:, d], method="pearson")
        oo_df.loc[c, d] = corr_cont

    oo_df2 = oo_df.stack().reset_index()
    oo_df2.columns = ["Predictor 1", "Predictor 2", "Correlation"]
    oo_df2 = oo_df2.sort_values(by=["Correlation"], ascending=[False])

    oo_df = oo_df.T
    fig2 = px.imshow(
        oo_df, labels=dict(x="Continuous Continuous Correlation", color="Correlation")
    )
    fig2.update_xaxes(side="top")


if __name__ == "__main__":
    sys.exit(__main__())
