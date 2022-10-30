import itertools
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import scipy.stats as ss

import dataset_loader

pd.options.mode.chained_assignment = None

pio.renderers.default = "browser"

"""
Sorry for the mess, skip to the end for more clear results.
"""

df, predictors, response = dataset_loader.get_test_data_set("titanic")

response = [response]
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

cont_cat_pair = [(a, b) for a in cont_col for b in cat_col]
cont_cont_pair = list(itertools.combinations(cont_col, 2))
cat_cat_pair = list(itertools.combinations(cat_col, 2))

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


""" Methods used from https://teaching.mrsharky.com/ """


def fill_na(df):
    if isinstance(df, pd.Series):
        return df.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in df])


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    """
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T

    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow

    Bias correction and formula's taken from : https://www.researchgate.net/publication/270277061_A_bias-correction_for
    _Cramer's_V_and_Tschuprow's_T

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = ss.chi2_contingency(crosstab_matrix, correction=yates_correct)
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


aa_df = pd.DataFrame(index=cat_col, columns=cat_col)

for j, k in cat_cat_pair:
    x = df.loc[:, j].astype("category").cat.codes
    y = df.loc[:, k].astype("category").cat.codes
    cat_correlation(x, y)
    aa_df.loc[j, k] = cat_correlation(x, y)


aa_df2 = aa_df.stack().reset_index()
aa_df2.columns = ["Predictor 1", "Predictor 2", "Correlation"]
aa_df2 = aa_df2.sort_values(by=["Correlation"], ascending=[False])


aa_df = aa_df.T
fig1 = px.imshow(
    aa_df, labels=dict(x="Categorical Categorical Correlation", color="Correlation")
)
fig1.update_xaxes(side="top")


def cat_cont_correlation_ratio(values, categories):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


ao_df = pd.DataFrame(index=cont_col, columns=cat_col)

for m, n in cont_cat_pair:
    mo = df.loc[:, m]
    no = df.loc[:, n].astype("category").cat.codes
    cat_correlation(mo, no)
    ao_df.loc[m, n] = cat_correlation(mo, no)


ao_df2 = ao_df.stack().reset_index()
ao_df2.columns = ["Predictor 1", "Predictor 2", "Correlation"]
ao_df2 = ao_df2.sort_values(by=["Correlation"], ascending=[False])


fig3 = px.imshow(
    ao_df, labels=dict(x="Continuous Categorical Correlation", color="Correlation")
)
fig3.update_xaxes(side="top")


for reset in df.columns:
    if reset in cat_col:
        df[reset] = df[reset].astype("category").cat.codes

resp_oo_df = {}
cont_cont_tuples = []

for item in cont_cont_pair:
    item = list(item)
    item.insert(0, "survived")
    cont_cont_tuples.append(tuple(item))

for items1 in cont_cont_tuples:
    resp_oo_df[items1] = pd.DataFrame()
    resp_oo_df[items1] = df.loc[:, items1]

resp_aa_df = {}
cat_cat_tuples = []

for item2 in cat_cat_pair:
    item2 = list(item2)
    item2.insert(0, "survived")
    cat_cat_tuples.append(tuple(item2))

for items2 in cat_cat_tuples:
    resp_aa_df[items2] = pd.DataFrame()
    resp_aa_df[items2] = df.loc[:, items2]

resp_ao_df = {}
cont_cat_tuples = []

for item3 in cont_cat_pair:
    item3 = list(item3)
    item3.insert(0, "survived")
    cont_cat_tuples.append(tuple(item3))

for items3 in cont_cat_tuples:
    resp_ao_df[items3] = pd.DataFrame()
    resp_ao_df[items3] = df.loc[:, items3]

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
    bin_diff = (tagsB > 0) & (tagsB < len(bins_B)) & (tagsA > 0) & (tagsA < len(bins_A))
    dfm = resp_oo_df[key].iloc[bin_diff]
    dfm["pred. 1"] = bins_A[(tagsA - 1)[bin_diff]]
    dfm["pred. 2"] = bins_B[(tagsB - 1)[bin_diff]]
    df_out = dfm.groupby(["pred. 1", "pred. 2"])["survived"].describe()
    cot_mse_dict[key] = df_out.iloc[:, [0, 1]]

cat_mse_dict = {}

for key, value in resp_aa_df.items():
    bins_A = np.arange(
        value.iloc[:, 1].min(), value.iloc[:, 1].max(), value.iloc[:, 1].max() / 5
    )
    bins_B = np.arange(
        value.iloc[:, 2].min(), value.iloc[:, 2].max(), value.iloc[:, 2].max() / 5
    )
    tagsA = np.searchsorted(bins_A, value.iloc[:, 1])
    tagsB = np.searchsorted(bins_B, value.iloc[:, 2])
    bin_diff = (
        (tagsB >= 0) & (tagsB < len(bins_B)) & (tagsA >= 0) & (tagsA < len(bins_A))
    )
    dfm = resp_aa_df[key].iloc[bin_diff]
    dfm["pred. 1"] = bins_A[(tagsA - 1)[bin_diff]]
    dfm["pred. 2"] = bins_B[(tagsB - 1)[bin_diff]]
    df2_out = dfm.groupby(["pred. 1", "pred. 2"])["survived"].describe()
    cat_mse_dict[key] = df2_out.iloc[:, [0, 1]]

caot_mse_dict = {}

for key, value in resp_ao_df.items():
    bins_A = np.arange(
        value.iloc[:, 1].min(), value.iloc[:, 1].max(), value.iloc[:, 1].max() / 5
    )
    bins_B = np.arange(
        value.iloc[:, 2].min(), value.iloc[:, 2].max(), value.iloc[:, 2].max() / 5
    )
    tagsA = np.searchsorted(bins_A, value.iloc[:, 1])
    tagsB = np.searchsorted(bins_B, value.iloc[:, 2])
    bin_diff = (
        (tagsB >= 0) & (tagsB < len(bins_B)) & (tagsA >= 0) & (tagsA < len(bins_A))
    )
    dfm = resp_ao_df[key].iloc[bin_diff]
    dfm["pred. 1"] = bins_A[(tagsA - 1)[bin_diff]]
    dfm["pred. 2"] = bins_B[(tagsB - 1)[bin_diff]]
    df3_out = dfm.groupby(["pred. 1", "pred. 2"])["survived"].describe()
    caot_mse_dict[key] = df3_out.iloc[:, [0, 1]]

pop_mean = df.survived.sum() / len(df.survived)

for key, value in caot_mse_dict.items():
    value["Bin Diff Mean"] = value.apply(
        lambda row: (pop_mean - row["mean"]) ** 2, axis=1
    )
for key, value in caot_mse_dict.items():
    value["Diff Mean Resp Pair"] = value["Bin Diff Mean"].sum() / 5

for key, value in cat_mse_dict.items():
    value["Bin Diff Mean"] = value.apply(
        lambda row: (pop_mean - row["mean"]) ** 2, axis=1
    )
for key, value in cat_mse_dict.items():
    value["Diff Mean Resp Pair"] = value["Bin Diff Mean"].sum() / 5

for key, value in cot_mse_dict.items():
    value["Bin Diff Mean"] = value.apply(
        lambda row: (pop_mean - row["mean"]) ** 2, axis=1
    )
for key, value in cot_mse_dict.items():
    value["Diff Mean Resp Pair"] = value["Bin Diff Mean"].sum() / 5

(
    "\n"
    "Apologies that this is not in a nice website. I couldn't figure all that out. \n"
    "\n"
    "Here are my 6 tables, with correlation matrices showing up in browser. \n"
    "\n"
)

# Categorical Categorical Correlation
print("Categorical Categorical Correlation: \n", aa_df2)

#  Continuous Categorical Correlation
print("\nContinuous Categorical Correlation: \n", ao_df2)

# Continuous Continuous Correlation
print("\nContinuous Continuous Correlation: \n", oo_df2)


# Continuous Continuous Correlation Matrix
fig2.show()

# Continuous Categorical Correlation Matrix
fig3.show()

# Categorical Categorical Correlation Matrix
fig1.show()

# Difference of Mean Response Continuous Continuous
for key, value in cot_mse_dict.items():
    print(key, "\n", value, "\n")

# Difference of Mean Response Continuous Categorical
for key, value in caot_mse_dict.items():
    print(key, "\n", value, "\n")

# Difference of Mean Response Categorical Categorical
for key, value in cat_mse_dict.items():
    print(key, "\n", value, "\n")
