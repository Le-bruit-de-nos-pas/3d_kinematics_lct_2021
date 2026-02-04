
# !pip install scikit_posthocs
# !pip install pingouin

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

print('Starting LCT_ClinicalData_ws.py')

# Read source CSV
try:
    LevodopaChallengeWideClinical = pd.read_csv('LevodopaChallengeWideClinical.csv', sep=';')
    print('Loaded LevodopaChallengeWideClinical:', LevodopaChallengeWideClinical.shape)
except Exception as e:
    print('Could not read LevodopaChallengeWideClinical.csv with sep=";":', e)
    try:
        LevodopaChallengeWideClinical = pd.read_csv('LevodopaChallengeWideClinical.csv')
        print('Loaded LevodopaChallengeWideClinical with default sep:', LevodopaChallengeWideClinical.shape)
    except Exception as e2:
        print('Failed to load LevodopaChallengeWideClinical.csv:', e2)
        LevodopaChallengeWideClinical = pd.DataFrame()

# Show head
print(LevodopaChallengeWideClinical.head())

# Visualize missingness (guard in case missingno fails)
try:
    mno.matrix(LevodopaChallengeWideClinical, figsize=(24,12))
    plt.show()
except Exception as e:
    print('missingno.matrix failed:', e)

# Remove patient index (first column)
if not LevodopaChallengeWideClinical.empty:
    patid_LevodopaChallengeWideClinical = LevodopaChallengeWideClinical.iloc[:,0:1]
else:
    patid_LevodopaChallengeWideClinical = pd.DataFrame()

# Select remaining, numeric variables (columns 1..45)
try:
    LevodopaChallengeWideClinical = LevodopaChallengeWideClinical.iloc[:,1:46]
except Exception as e:
    print('Selecting columns 1:46 failed:', e)

print('After selection, shape:', LevodopaChallengeWideClinical.shape)

# List missing columns
missing_columns = list(LevodopaChallengeWideClinical.columns)
print('missing_columns:', missing_columns)

# Create _imp copies and try random_imputation if present
for feature in missing_columns:
    LevodopaChallengeWideClinical[feature + '_imp'] = LevodopaChallengeWideClinical[feature]
    try:
        LevodopaChallengeWideClinical = random_imputation(LevodopaChallengeWideClinical, feature)
    except NameError:
        # random_imputation not defined; skipping imputation
        pass
    except Exception as e:
        print(f'random_imputation failed for {feature}:', e)

# Estimate missing data using a regression model
if missing_columns:
    deter_data = pd.DataFrame(columns=["Det" + name for name in missing_columns])
    for feature in missing_columns:
        try:
            deter_data["Det" + feature] = LevodopaChallengeWideClinical[feature + "_imp"]
            parameters = list(set(LevodopaChallengeWideClinical.columns) - set(missing_columns) - {feature + '_imp'})
            model = linear_model.LinearRegression()
            # If parameters empty this will fail; guard
            if parameters:
                model.fit(X=LevodopaChallengeWideClinical[parameters], y=LevodopaChallengeWideClinical[feature + '_imp'])
                mask = LevodopaChallengeWideClinical[feature].isnull()
                if mask.any():
                    deter_data.loc[mask, "Det" + feature] = model.predict(LevodopaChallengeWideClinical[parameters])[mask]
            else:
                # No parameters to fit; copy the imp column
                deter_data["Det" + feature] = LevodopaChallengeWideClinical[feature + "_imp"]
        except Exception as e:
            print('Regression estimate failed for', feature, e)
else:
    deter_data = pd.DataFrame()

# Visualize deter_data missingness
try:
    if not deter_data.empty:
        mno.matrix(deter_data, figsize=(24,12))
        plt.show()
except Exception as e:
    print('missingno on deter_data failed:', e)

# Save to CSV (guarded)
try:
    if not deter_data.empty:
        deter_data.to_csv('output_wide_clinical.csv', encoding='utf-8-sig', index=False)
        # files.download('output_wide_clinical.csv')  # Notebook-only
        print('Wrote output_wide_clinical.csv')
except Exception as e:
    print('Failed to write output_wide_clinical.csv:', e)

# Read output_wide_clinical (note some notebooks use sep=',' vs ';')
output_wide_clinical = pd.DataFrame()
for sep in [';', ',']:
    try:
        output_wide_clinical = pd.read_csv('output_wide_clinical.csv', sep=sep)
        print('Loaded output_wide_clinical with sep', sep, 'shape', output_wide_clinical.shape)
        break
    except Exception:
        continue
if output_wide_clinical.empty:
    print('output_wide_clinical.csv not found or empty; proceeding with empty DataFrame')

print(output_wide_clinical.head())

# Quick utility to run repeated blocks (Friedman + posthoc + violin plot)
def run_group_stats_and_plot(df, var, group_col='Group', subject_col='patient', violin_color='midnightblue', swarm_size=10, figsize=(8,6), xticklabels=None):
    if df is None or df.empty or var not in df.columns:
        print(f'Skipping {var}: not present')
        return
    try:
        print('\n' + var + ' summary:')
        print(df.groupby(group_col).describe()[var])
    except Exception as e:
        print('Describe failed for', var, e)
    try:
        sns.set(style='white')
        ax = sns.violinplot(x=group_col, y=var, data=df, color=violin_color, linewidth=0, alpha=1, scale='width', bw=0.2, cut=2)
        sns.swarmplot(y=var, x=group_col, data=df, color=violin_color, edgecolor=violin_color, size=swarm_size, alpha=1)
        ax.set(xlabel=None)
        ax.set(ylabel='\n ' + var + ' \n')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
        if xticklabels:
            ax.set_xticklabels(xticklabels)
        sns.set(rc={'figure.figsize':figsize})
        sns.despine(left=True, bottom=True)
        plt.show()
    except Exception as e:
        print('Plot failed for', var, e)
    try:
        print(pg.friedman(data=df, dv=var, within=group_col, subject=subject_col))
    except Exception as e:
        print('Friedman failed for', var, e)
    try:
        print(sp.posthoc_conover_friedman(a=df, y_col=var, group_col=group_col, block_col=subject_col, p_adjust='fdr_bh', melted=True))
    except Exception as e:
        print('Posthoc Conover failed for', var, e)

# If output_wide_clinical has expected columns, run through a few known variables
vars_to_check = ['Short_MDS-UPDRS_III', 'Axial_Score', 'Item_3.1', 'Item_3.3', 'Item_3.4', 'Item_3.8', 'Item_3.10', 'Item_3.11', 'Item_3.12', 'Item_3.15', 'Item_3.17']
for v in vars_to_check:
    run_group_stats_and_plot(output_wide_clinical, v, xticklabels=['OFF', '20 min', '40 min', '60 min', '80 min'])

# OFF vs Best ON section
try:
    BestON_output_wide_clinical = None
    for sep in [';', ',']:
        try:
            BestON_output_wide_clinical = pd.read_csv('BestON_output_wide_clinical.csv', sep=sep)
            print('Loaded BestON_output_wide_clinical with sep', sep, 'shape', BestON_output_wide_clinical.shape)
            break
        except Exception:
            continue
    if BestON_output_wide_clinical is None or BestON_output_wide_clinical.empty:
        print('BestON_output_wide_clinical.csv not found or empty')
    else:
        # MDS-UPDRS III
        try:
            print(BestON_output_wide_clinical[['MDS_UPDRS_III_OFF','MDS_UPDRS_III_BestON']].describe())
            print(stats.wilcoxon(BestON_output_wide_clinical['MDS_UPDRS_III_OFF'], BestON_output_wide_clinical['MDS_UPDRS_III_BestON']))
            df1 = BestON_output_wide_clinical[['patient', 'MDS_UPDRS_III_OFF','MDS_UPDRS_III_BestON']]
            df1 = df1.set_index('patient').stack().reset_index()
            df1.columns = ['patient','Group','UPDRS_III']
            sns.set(style='white')
            ax = sns.violinplot(x='Group', y='UPDRS_III', data=df1, color='khaki', linewidth=0, alpha=1, scale='width', bw=0.2, cut=2)
            sns.swarmplot(y='UPDRS_III', x='Group', data=df1, color='olivedrab', edgecolor='olivedrab', size=10, alpha=1)
            ax.set(xlabel=None)
            ax.set(ylabel='\n MDS-UPDRS III \n')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_xticklabels(['OFF', 'Best ON'])
            sns.set(rc={'figure.figsize':(5,5)})
            sns.despine(left=True, bottom=True)
            plt.show()
        except Exception as e:
            print('BestON MDS-UPDRS block failed:', e)

        # Other scores: Axial, Tremor, Rigidity, Akinesia
        paired_vars = [('Axial_OFF','Axial_BestON','Axial Score'), ('Tremor_OFF','Tremor_BestON','Tremor Score'), ('Rigidity_OFF','Rigidity_BestON','Rigidity Score'), ('Akinesia_OFF','Akinesia_BestON','Akinesia Score')]
        for off_col, on_col, pretty in paired_vars:
            try:
                print(BestON_output_wide_clinical[[off_col, on_col]].describe())
                print(stats.wilcoxon(BestON_output_wide_clinical[off_col], BestON_output_wide_clinical[on_col]))
                df1 = BestON_output_wide_clinical[['patient', off_col, on_col]]
                df1 = df1.set_index('patient').stack().reset_index()
                df1.columns = ['patient','Group', pretty]
                sns.set(style='white')
                ax = sns.violinplot(x='Group', y=pretty, data=df1, color='khaki', linewidth=0, alpha=1, scale='width', bw=0.2, cut=2)
                sns.swarmplot(y=pretty, x='Group', data=df1, color='olivedrab', edgecolor='olivedrab', size=10, alpha=1)
                ax.set(xlabel=None)
                ax.set(ylabel='\n ' + pretty + ' \n')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.set_xticklabels(['OFF', 'Best ON'])
                sns.set(rc={'figure.figsize':(5,5)})
                sns.despine(left=True, bottom=True)
                plt.show()
            except Exception as e:
                print(f'BestON paired block failed for {pretty}:', e)

except Exception as e:
    print('OFF vs BestON block failed overall:', e)

print('\nConversion finished: LCT_ClinicalData_ws.py')
