
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
import sys
from math import sqrt

# from google.colab import files

from sklearn import linear_model
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats

# Read data
output_wide_BestON_OFF = pd.read_csv('output_wide_BestON_OFF.csv', sep=';', decimal='.')

# Trim whitespace
output_wide_BestON_OFF = output_wide_BestON_OFF.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Helpers
def p_rounder(p_value):
    if p_value < .0001:
        p_value = '<.0001'
    else:
        p_value = str((round(p_value,4)))
    return p_value


def bon_correct(p_value,k):
    corrected_p = p_value * ((k *(k-1))/2)
    return p_value, corrected_p


def kw_dunn_post_hoc(df,strat,comp_list, var):
    post_hoc_result_dict = {}
    N = df['rank'].count()
    n_groups = df[strat].nunique()
    for comp in comp_list:
        m1 = df.loc[df[strat] == comp[0]]['rank'].mean()
        n1 = df.loc[df[strat] == comp[0]]['rank'].count()
        m2 = df.loc[df[strat] == comp[1]]['rank'].mean()
        n2 = df.loc[df[strat] == comp[1]]['rank'].count()
        Z = (m1 - m2)/sqrt(((N*(N+1))/12)*((1/n1)+(1/n2)))
        Z = round(Z,4)
        p = stats.norm.sf(abs(Z))
        p, corrected_p = bon_correct(p,n_groups)
        p = p_rounder(p)
        corrected_p = p_rounder(corrected_p)
        comparison = f'{comp[0]} vs. {comp[1]}'
        post_hoc_result_dict[comparison] = [var,Z,p,corrected_p]
    return post_hoc_result_dict


def kw_test(df,stratifier,var):
    result_list = []
    strat_list = []
    comparison_list = []
    counter = 0
    temp_df = df[[stratifier,var]].copy()
    temp_df['rank'] = temp_df[var].rank(method='average')
    for strat in df[stratifier].unique():
        result = df.loc[df[stratifier] == strat][var].values
        result_list.append(result)
        strat_list.append(strat)
    for st in strat_list:
        for st2 in strat_list:
            if st != st2 and [st2,st] not in comparison_list:
                comparison_list.append([st,st2])
    post_hoc_result_dict = kw_dunn_post_hoc(temp_df,stratifier,comparison_list,var)
    if len(result_list) == 2:
        k,p = stats.kruskal(result_list[0],result_list[1])
    if len(result_list) == 3:
        k,p = stats.kruskal(result_list[0],result_list[1],result_list[2])
    elif len(result_list) == 4:
        k,p = stats.kruskal(result_list[0],result_list[1],result_list[2],result_list[3])
    elif len(result_list) == 5:
        k,p = stats.kruskal(result_list[0],result_list[1],result_list[2],result_list[3],result_list[4])
    else:
        print('Stratifying levels greater than 5. Please modify code to accomodate.')
        sys.exit()
    k = round(k,4)    
    p = p_rounder(p)
    return k, p, post_hoc_result_dict


# Process every numeric variable automatically:
def process_and_plot(var, df, group_col='Group', violin_color='lightcoral', swarm_color='darkred', figsize=(6,6), swarm_size=8):
    """Run kruskal/ post-hoc and plot violin+swarm for a single variable.

    This replaces the repeated per-variable code blocks in the original notebook and
    ensures every numeric column is processed.
    """
    try:
        if var not in df.columns:
            return
        # skip non-numeric or all-NaN columns
        if not pd.api.types.is_numeric_dtype(df[var]):
            return
        if df[var].dropna().shape[0] == 0:
            return

        print(f"--- {var} ---")
        try:
            print(df.groupby(group_col).describe()[var])
        except Exception:
            # Some columns may have characters; ignore describe errors
            pass

        # run Kruskal-Wallis + post-hoc
        try:
            k, p, post = kw_test(df, group_col, var)
            print('Kruskal-Wallis:', k, 'p:', p)
            # print post-hoc summary
            for comp, vals in post.items():
                print(comp, vals)
        except Exception as e:
            print('kw_test failed for', var, '->', e)

        # Plot
        try:
            sns.set(style='white')
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax = sns.violinplot(x=group_col, y=var, data=df, color=violin_color, linewidth=0, alpha=1, scale='width', bw=0.2, cut=2)
            sns.swarmplot(y=var, x=group_col, data=df, color=swarm_color, edgecolor=swarm_color, size=swarm_size, alpha=1)
            ax.set(xlabel=None)
            ax.set(ylabel=f"\n {var} \n")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
            # Try to preserve label order if possible
            try:
                ax.set_xticklabels(['OFF', 'Best ON', 'Control'])
            except Exception:
                pass
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print('Plot failed for', var, '->', e)
    except Exception as e:
        print('Unexpected error processing', var, '->', e)


# Build list of numeric variables to process (exclude the group column)
all_vars = [c for c in output_wide_BestON_OFF.columns if c != 'Group']

for var in all_vars:
    process_and_plot(var, output_wide_BestON_OFF, group_col='Group')

# --- t-SNE section ---
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.manifold import _t_sne

# Re-read CSV without decimal specification for t-SNE section
output_wide_BestON_OFF = pd.read_csv('output_wide_BestON_OFF.csv', sep=';')
# Keep only groups not OFF
output_wide_BestON_OFF = output_wide_BestON_OFF.loc[output_wide_BestON_OFF['Group'] != 'OFF']

# Separate group labels and features (original notebook used columns 1:65 etc.)
groups_output_wide_BestON_OFF = output_wide_BestON_OFF.iloc[:,65:66]
output_wide_BestON_OFF = output_wide_BestON_OFF.iloc[:,1:65]

distance_matrix_sorted = pairwise_distances(output_wide_BestON_OFF, metric='euclidean')
fig, ax = plt.subplots(1,1)
ax.imshow(distance_matrix_sorted, 'Greys')
ax.set_title("Sorted by Label")
sns.set(rc={'figure.figsize':(14,14)})

perplexity = 30  # Same as the default perplexity
p = _t_sne._joint_probabilities(distances=distance_matrix_sorted, desired_perplexity = perplexity, verbose=False)

n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(output_wide_BestON_OFF)
print(tsne_result.shape)

groups_output_wide_BestON_OFF = groups_output_wide_BestON_OFF.to_numpy()
groups_output_wide_BestON_OFF = groups_output_wide_BestON_OFF.ravel()

# Quick scatter plot of t-SNE result
sns.set(style="white")
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': groups_output_wide_BestON_OFF})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, palette="BuPu", style="label", ax=ax,s=120)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
sns.set(rc={'figure.figsize':(5,5)})
sns.despine(left=True, bottom=True)

# --- Optional: optimized embedding via gradient descent (advanced, uses private _t_sne internals) ---
# Create the initial embedding
n_samples = output_wide_BestON_OFF.shape[0]
n_components = 2
X_embedded = 1e-4 * np.random.randn(n_samples, n_components).astype(np.float32)
embedding_init = X_embedded.ravel()  # Flatten the two dimensional array to 1D

# kl_kwargs for internal _kl_divergence
kl_kwargs = {'P': p, 'degrees_of_freedom': 1, 'n_samples': n_samples, 'n_components':2}

# Perform gradient descent
embedding_done = _t_sne._gradient_descent(_t_sne._kl_divergence, embedding_init, 0, n_samples, kwargs=kl_kwargs)

# Get first and second TSNE components into a 2D array
tsne_result = embedding_done[0].reshape(n_samples,2)

# Convert to DataFrame and plot
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': groups_output_wide_BestON_OFF})
sns.set(style="white")
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette="BuPu" , style="label", data=tsne_result_df, ax=ax,s=120)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
sns.set(rc={'figure.figsize':(8,8)})
sns.despine(left=True, bottom=True)

print('Script finished.')
