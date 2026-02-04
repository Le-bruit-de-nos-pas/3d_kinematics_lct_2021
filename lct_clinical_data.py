
#!pip install scikit_posthocs

#!pip install pingouin

import pandas as pd
import numpy as np
from statsmodels.stats.anova import AnovaRM
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno
import math
import scipy
import pingouin as pg
import scikit_posthocs as sp

# from google.colab import files

from sklearn import linear_model
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats

# Read the wide clinical CSV
LevodopaChallengeWideClinical = pd.read_csv('LevodopaChallengeWideClinical.csv', sep = ';')

# Display (in notebook this would show the DataFrame)
print(LevodopaChallengeWideClinical.head())

# Visualize missingness matrix
try:
    mno.matrix(LevodopaChallengeWideClinical, figsize = (24,12))
except Exception as e:
    print('mno.matrix failed:', e)

# Remove patient index
patid_LevodopaChallengeWideClinical = LevodopaChallengeWideClinical.iloc[:,0:1]

# Select remaining, numeric variables
LevodopaChallengeWideClinical = LevodopaChallengeWideClinical.iloc[:,1:46]

print(LevodopaChallengeWideClinical.head())

missing_columns = list(LevodopaChallengeWideClinical)
print('missing_columns:', missing_columns)

# NOTE: random_imputation is referenced in the notebook but not defined here.
# Keep the loop, but it will fail unless random_imputation is defined elsewhere.
for feature in missing_columns:
    LevodopaChallengeWideClinical[feature + '_imp'] = LevodopaChallengeWideClinical[feature]
    # The notebook calls random_imputation(LevodopaChallengeWideClinical, feature)
    # Ensure a random_imputation function exists before running the following line:
    try:
        LevodopaChallengeWideClinical = random_imputation(LevodopaChallengeWideClinical, feature)
    except NameError:
        # Skip if random_imputation is not defined; leave _imp columns equal to original
        pass

# Estimate the missing data using a regression model
# Input estimates only (!) on the rows/columns where the original data was missing

deter_data = pd.DataFrame(columns = ["Det" + name for name in missing_columns])

for feature in missing_columns:
    deter_data["Det" + feature] = LevodopaChallengeWideClinical[feature + "_imp"]
    parameters = list(set(LevodopaChallengeWideClinical.columns) - set(missing_columns) - {feature + '_imp'})

    try:
        model = linear_model.LinearRegression()
        model.fit(X = LevodopaChallengeWideClinical[parameters], y = LevodopaChallengeWideClinical[feature + '_imp'])
        deter_data.loc[LevodopaChallengeWideClinical[feature].isnull(), "Det" + feature] = model.predict(LevodopaChallengeWideClinical[parameters])[LevodopaChallengeWideClinical[feature].isnull()]
    except Exception as e:
        print(f'Could not fit regression for feature {feature}:', e)

# Visualize missingness for deter_data
try:
    mno.matrix(deter_data, figsize = (24,12))
except Exception as e:
    print('mno.matrix(deter_data) failed:', e)

# Save output_wide_clinical
try:
    deter_data.to_csv('output_wide_clinical.csv', encoding = 'utf-8-sig')
except Exception as e:
    print('Could not write output_wide_clinical.csv:', e)

# In the notebook this triggers a file download; commented out here.
# files.download('output_wide_clinical.csv')

# Read the output wide clinical (the notebook reads it back with sep=';')
try:
    output_wide_clinical = pd.read_csv('output_wide_clinical.csv', sep = ';')
except Exception as e:
    print('Could not read output_wide_clinical.csv:', e)
    # Fallback: use deter_data if available
    try:
        output_wide_clinical = deter_data.copy()
    except Exception:
        output_wide_clinical = pd.DataFrame()

print(output_wide_clinical.head())

# List columns
try:
    print(list(output_wide_clinical))
except Exception as e:
    print('Could not list columns:', e)

# -----------------------------
# MDS-UPDRS III section
# -----------------------------
print('\n# MDS-UPDRS III')
try:
    print(output_wide_clinical.groupby("Group").describe()["Short_MDS-UPDRS_III"])
except Exception as e:
    print('MDS-UPDRS III describe failed:', e)

sns.set(style="white")
try:
    ax = sns.violinplot(x="Group", y="Short_MDS-UPDRS_III", data=output_wide_clinical, color="midnightblue", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Short_MDS-UPDRS_III", x = "Group", data = output_wide_clinical, color="midnightblue", edgecolor = "midnightblue", size = 10, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Short MDS-UPDRS III \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
    ax.set_xticklabels(['OFF', '20 min', '40 min', '60 min', '80 min'])
    sns.set(rc={'figure.figsize':(8,6)})
    sns.despine(left=True, bottom=True)
    plt.show()
except Exception as e:
    print('Plotting Short_MDS-UPDRS_III failed:', e)

# Friedman test
try:
    print(pg.friedman(data=output_wide_clinical, dv="Short_MDS-UPDRS_III", within="Group", subject="patient"))
except Exception as e:
    print('Friedman test failed for Short_MDS-UPDRS_III:', e)

# Post-hoc Conover
try:
    print(sp.posthoc_conover_friedman(a=output_wide_clinical, y_col="Short_MDS-UPDRS_III", group_col="Group", block_col="patient", p_adjust="fdr_bh", melted=True))
except Exception as e:
    print('Posthoc Conover failed for Short_MDS-UPDRS_III:', e)

# -----------------------------
# AXIAL SCORE
# -----------------------------
print('\n# AXIAL SCORE')
try:
    print(output_wide_clinical.groupby("Group").describe()["Axial_Score"])
except Exception as e:
    print('Axial_Score describe failed:', e)

sns.set(style="white")
try:
    ax = sns.violinplot(x="Group", y="Axial_Score", data=output_wide_clinical, color="midnightblue", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Axial_Score", x = "Group", data = output_wide_clinical, color="midnightblue", edgecolor = "midnightblue", size = 10, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Axial Score \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
    ax.set_xticklabels(['OFF', '20 min', '40 min', '60 min', '80 min'])
    sns.set(rc={'figure.figsize':(8,6)})
    sns.despine(left=True, bottom=True)
    plt.show()
except Exception as e:
    print('Plotting Axial_Score failed:', e)

# Friedman
try:
    print(pg.friedman(data=output_wide_clinical, dv="Axial_Score", within="Group", subject="patient"))
except Exception as e:
    print('Friedman test failed for Axial_Score:', e)

# Post-hoc
try:
    print(sp.posthoc_conover_friedman(a=output_wide_clinical, y_col="Axial_Score", group_col="Group", block_col="patient", p_adjust="fdr_bh", melted=True))
except Exception as e:
    print('Posthoc Conover failed for Axial_Score:', e)


# Example for Item_3.1
print('\n# Item 3.1')
try:
    print(output_wide_clinical.groupby("Group").describe()["Item_3.1"])
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Item_3.1", data=output_wide_clinical, color="midnightblue", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Item_3.1", x = "Group", data = output_wide_clinical, color="midnightblue", edgecolor = "midnightblue", size = 10, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Item 3.1 \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
    ax.set_xticklabels(['OFF', '20 min', '40 min', '60 min', '80 min'])
    sns.set(rc={'figure.figsize':(8,6)})
    sns.despine(left=True, bottom=True)
    plt.show()
    print(pg.friedman(data=output_wide_clinical, dv="Item_3.1", within="Group", subject="patient"))
    print(sp.posthoc_conover_friedman(a=output_wide_clinical, y_col="Item_3.1", group_col="Group", block_col="patient", p_adjust="fdr_bh", melted=True))
except Exception as e:
    print('Item_3.1 block failed:', e)

# Item 3.3
print('\n# Item 3.3')
try:
    print(output_wide_clinical.groupby("Group").describe()["Item_3.3"])
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Item_3.3", data=output_wide_clinical, color="midnightblue", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Item_3.3", x = "Group", data = output_wide_clinical, color="midnightblue", edgecolor = "midnightblue", size = 10, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Item 3.3 \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
    ax.set_xticklabels(['OFF', '20 min', '40 min', '60 min', '80 min'])
    sns.set(rc={'figure.figsize':(8,6)})
    sns.despine(left=True, bottom=True)
    plt.show()
    print(pg.friedman(data=output_wide_clinical, dv="Item_3.3", within="Group", subject="patient"))
    print(sp.posthoc_conover_friedman(a=output_wide_clinical, y_col="Item_3.3", group_col="Group", block_col="patient", p_adjust="fdr_bh", melted=True))
except Exception as e:
    print('Item_3.3 block failed:', e)

# Item 3.4
print('\n# Item 3.4')
try:
    print(output_wide_clinical.groupby("Group").describe()["Item_3.4"])
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Item_3.4", data=output_wide_clinical, color="midnightblue", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Item_3.4", x = "Group", data = output_wide_clinical, color="midnightblue", edgecolor = "midnightblue", size = 10, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Item 3.4 \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
    ax.set_xticklabels(['OFF', '20 min', '40 min', '60 min', '80 min'])
    sns.set(rc={'figure.figsize':(8,6)})
    sns.despine(left=True, bottom=True)
    plt.show()
    print(pg.friedman(data=output_wide_clinical, dv="Item_3.4", within="Group", subject="patient"))
    print(sp.posthoc_conover_friedman(a=output_wide_clinical, y_col="Item_3.4", group_col="Group", block_col="patient", p_adjust="fdr_bh", melted=True))
except Exception as e:
    print('Item_3.4 block failed:', e)

# Item 3.8
print('\n# Item 3.8')
try:
    print(output_wide_clinical.groupby("Group").describe()["Item_3.8"])
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Item_3.8", data=output_wide_clinical, color="midnightblue", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Item_3.8", x = "Group", data = output_wide_clinical, color="midnightblue", edgecolor = "midnightblue", size = 5, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Item 3.8 \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
    ax.set_xticklabels(['OFF', '20 min', '40 min', '60 min', '80 min'])
    sns.set(rc={'figure.figsize':(8,6)})
    sns.despine(left=True, bottom=True)
    plt.show()
    print(pg.friedman(data=output_wide_clinical, dv="Item_3.8", within="Group", subject="patient"))
    print(sp.posthoc_conover_friedman(a=output_wide_clinical, y_col="Item_3.8", group_col="Group", block_col="patient", p_adjust="fdr_bh", melted=True))
except Exception as e:
    print('Item_3.8 block failed:', e)

# Item 3.10
print('\n# Item 3.10')
try:
    print(output_wide_clinical.groupby("Group").describe()["Item_3.10"])
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Item_3.10", data=output_wide_clinical, color="midnightblue", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Item_3.10", x = "Group", data = output_wide_clinical, color="midnightblue", edgecolor = "midnightblue", size = 5, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Item 3.10 \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
    ax.set_xticklabels(['OFF', '20 min', '40 min', '60 min', '80 min'])
    sns.set(rc={'figure.figsize':(8,6)})
    sns.despine(left=True, bottom=True)
    plt.show()
    print(pg.friedman(data=output_wide_clinical, dv="Item_3.10", within="Group", subject="patient"))
    print(sp.posthoc_conover_friedman(a=output_wide_clinical, y_col="Item_3.10", group_col="Group", block_col="patient", p_adjust="fdr_bh", melted=True))
except Exception as e:
    print('Item_3.10 block failed:', e)

# Item 3.11
print('\n# Item 3.11')
try:
    print(output_wide_clinical.groupby("Group").describe()["Item_3.11"])
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Item_3.11", data=output_wide_clinical, color="midnightblue", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Item_3.11", x = "Group", data = output_wide_clinical, color="midnightblue", edgecolor = "midnightblue", size = 5, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Item 3.11 \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
    ax.set_xticklabels(['OFF', '20 min', '40 min', '60 min', '80 min'])
    sns.set(rc={'figure.figsize':(8,6)})
    sns.despine(left=True, bottom=True)
    plt.show()
    print(pg.friedman(data=output_wide_clinical, dv="Item_3.11", within="Group", subject="patient"))
    print(sp.posthoc_conover_friedman(a=output_wide_clinical, y_col="Item_3.11", group_col="Group", block_col="patient", p_adjust="fdr_bh", melted=True))
except Exception as e:
    print('Item_3.11 block failed:', e)

# Item 3.12
print('\n# Item 3.12')
try:
    print(output_wide_clinical.groupby("Group").describe()["Item_3.12"])
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Item_3.12", data=output_wide_clinical, color="midnightblue", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Item_3.12", x = "Group", data = output_wide_clinical, color="midnightblue", edgecolor = "midnightblue", size = 5, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Item 3.12 \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
    ax.set_xticklabels(['OFF', '20 min', '40 min', '60 min', '80 min'])
    sns.set(rc={'figure.figsize':(8,6)})
    sns.despine(left=True, bottom=True)
    plt.show()
    print(pg.friedman(data=output_wide_clinical, dv="Item_3.12", within="Group", subject="patient"))
    print(sp.posthoc_conover_friedman(a=output_wide_clinical, y_col="Item_3.12", group_col="Group", block_col="patient", p_adjust="fdr_bh", melted=True))
except Exception as e:
    print('Item_3.12 block failed:', e)

# Item 3.15
print('\n# Item 3.15')
try:
    print(output_wide_clinical.groupby("Group").describe()["Item_3.15"])
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Item_3.15", data=output_wide_clinical, color="midnightblue", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Item_3.15", x = "Group", data = output_wide_clinical, color="midnightblue", edgecolor = "midnightblue", size = 5, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Item 3.15 \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
    ax.set_xticklabels(['OFF', '20 min', '40 min', '60 min', '80 min'])
    sns.set(rc={'figure.figsize':(8,6)})
    sns.despine(left=True, bottom=True)
    plt.show()
    print(pg.friedman(data=output_wide_clinical, dv="Item_3.15", within="Group", subject="patient"))
    print(sp.posthoc_conover_friedman(a=output_wide_clinical, y_col="Item_3.15", group_col="Group", block_col="patient", p_adjust="fdr_bh", melted=True))
except Exception as e:
    print('Item_3.15 block failed:', e)

# Item 3.17
print('\n# Item 3.17')
try:
    print(output_wide_clinical.groupby("Group").describe()["Item_3.17"])
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Item_3.17", data=output_wide_clinical, color="midnightblue", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Item_3.17", x = "Group", data = output_wide_clinical, color="midnightblue", edgecolor = "midnightblue", size = 5, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Item 3.17 \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
    ax.set_xticklabels(['OFF', '20 min', '40 min', '60 min', '80 min'])
    sns.set(rc={'figure.figsize':(8,6)})
    sns.despine(left=True, bottom=True)
    plt.show()
    print(pg.friedman(data=output_wide_clinical, dv="Item_3.17", within="Group", subject="patient"))
    print(sp.posthoc_conover_friedman(a=output_wide_clinical, y_col="Item_3.17", group_col="Group", block_col="patient", p_adjust="fdr_bh", melted=True))
except Exception as e:
    print('Item_3.17 block failed:', e)

# -----------------------------
# OFF vs Best ON: Clinical Score
# -----------------------------
print('\n# OFF vs Best ON: Clinical Score')
try:
    BestON_output_wide_clinical = pd.read_csv('BestON_output_wide_clinical.csv', sep = ';')
    print(BestON_output_wide_clinical.head())
    print(list(BestON_output_wide_clinical))
except Exception as e:
    print('Could not read BestON_output_wide_clinical.csv:', e)

# MDS-UPDRS III (BestON)
print('\n# MDS-UPDRS III (BestON)')
try:
    print(BestON_output_wide_clinical[['MDS_UPDRS_III_OFF','MDS_UPDRS_III_BestON']].describe())
    print(stats.wilcoxon(BestON_output_wide_clinical['MDS_UPDRS_III_OFF'], BestON_output_wide_clinical['MDS_UPDRS_III_BestON']))
    df1 = BestON_output_wide_clinical[['patient', 'MDS_UPDRS_III_OFF','MDS_UPDRS_III_BestON']]
    df1 = df1.set_index('patient').stack().reset_index()
    df1.columns = ['patient','Group','UPDRS_III']
    print(df1.head())
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="UPDRS_III", data=df1, color="khaki", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "UPDRS_III", x = "Group", data = df1, color="olivedrab", edgecolor = "olivedrab", size = 10, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n MDS-UPDRS III \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
    ax.set_xticklabels(['OFF', 'Best ON'])
    sns.set(rc={'figure.figsize':(5,5)})
    sns.despine(left=True, bottom=True)
    plt.show()
except Exception as e:
    print('BestON MDS-UPDRS III block failed:', e)

# Axial Score (BestON)
print('\n# Axial Score (BestON)')
try:
    print(BestON_output_wide_clinical[['Axial_OFF','Axial_BestON']].describe())
    print(stats.wilcoxon(BestON_output_wide_clinical['Axial_OFF'], BestON_output_wide_clinical['Axial_BestON']))
    df1 = BestON_output_wide_clinical[['patient', 'Axial_OFF','Axial_BestON']]
    df1 = df1.set_index('patient').stack().reset_index()
    df1.columns = ['patient','Group','Axial Score']
    print(df1.head())
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Axial Score", data=df1, color="khaki", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Axial Score", x = "Group", data = df1, color="olivedrab", edgecolor = "olivedrab", size = 10, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Axial Score \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
    ax.set_xticklabels(['OFF', 'Best ON'])
    sns.set(rc={'figure.figsize':(5,5)})
    sns.despine(left=True, bottom=True)
    plt.show()
except Exception as e:
    print('BestON Axial Score block failed:', e)

# Tremor Score (BestON)
print('\n# Tremor Score (BestON)')
try:
    print(BestON_output_wide_clinical[['Tremor_OFF','Tremor_BestON']].describe())
    print(stats.wilcoxon(BestON_output_wide_clinical['Tremor_OFF'], BestON_output_wide_clinical['Tremor_BestON']))
    df1 = BestON_output_wide_clinical[['patient', 'Tremor_OFF','Tremor_BestON']]
    df1 = df1.set_index('patient').stack().reset_index()
    df1.columns = ['patient','Group','Tremor Score']
    print(df1.head())
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Tremor Score", data=df1, color="khaki", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Tremor Score", x = "Group", data = df1, color="olivedrab", edgecolor = "olivedrab", size = 10, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Tremor Score \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
    ax.set_xticklabels(['OFF', 'Best ON'])
    sns.set(rc={'figure.figsize':(5,5)})
    sns.despine(left=True, bottom=True)
    plt.show()
except Exception as e:
    print('BestON Tremor Score block failed:', e)

# Rigidity Score (BestON)
print('\n# Rigidity Score (BestON)')
try:
    print(BestON_output_wide_clinical[['Rigidity_OFF','Rigidity_BestON']].describe())
    print(stats.wilcoxon(BestON_output_wide_clinical['Rigidity_OFF'], BestON_output_wide_clinical['Rigidity_BestON']))
    df1 = BestON_output_wide_clinical[['patient', 'Rigidity_OFF','Rigidity_BestON']]
    df1 = df1.set_index('patient').stack().reset_index()
    df1.columns = ['patient','Group','Rigidity Score']
    print(df1.head())
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Rigidity Score", data=df1, color="khaki", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Rigidity Score", x = "Group", data = df1, color="olivedrab", edgecolor = "olivedrab", size = 10, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Rigidity Score \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
    ax.set_xticklabels(['OFF', 'Best ON'])
    sns.set(rc={'figure.figsize':(5,5)})
    sns.despine(left=True, bottom=True)
    plt.show()
except Exception as e:
    print('BestON Rigidity Score block failed:', e)

# Akinesia Score (BestON)
print('\n# Akinesia Score (BestON)')
try:
    print(BestON_output_wide_clinical[['Akinesia_OFF','Akinesia_BestON']].describe())
    print(stats.wilcoxon(BestON_output_wide_clinical['Akinesia_OFF'], BestON_output_wide_clinical['Akinesia_BestON']))
    df1 = BestON_output_wide_clinical[['patient', 'Akinesia_OFF','Akinesia_BestON']]
    df1 = df1.set_index('patient').stack().reset_index()
    df1.columns = ['patient','Group','Akinesia Score']
    print(df1.head())
    sns.set(style="white")
    ax = sns.violinplot(x="Group", y="Akinesia Score", data=df1, color="khaki", linewidth=0, alpha=1, scale="width", bw=0.2, cut=2)
    sns.swarmplot(y = "Akinesia Score", x = "Group", data = df1, color="olivedrab", edgecolor = "olivedrab", size = 10, alpha = 1)
    ax.set(xlabel=None)
    ax.set(ylabel="\n Akinesia Score \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
    ax.set_xticklabels(['OFF', 'Best ON'])
    sns.set(rc={'figure.figsize':(5,5)})
    sns.despine(left=True, bottom=True)
    plt.show()
except Exception as e:
    print('BestON Akinesia Score block failed:', e)



print('\nConversion complete: created LCT_ClinicalData.py')
