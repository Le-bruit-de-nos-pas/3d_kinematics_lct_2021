"""
LCT_ALLTimePoints
-----------------

Analysis of LevodopaChallengeWide.csv:
- Missing-value imputation (random + linear-regression based).
- Export of imputed wide dataset.
- Descriptive statistics by Group.
- Violin + swarm plots for multiple gait variables.
- Friedman tests and Conover post-hoc tests for repeated measures.

Requirements (install via pip):
    pandas
    numpy
    statsmodels
    matplotlib
    seaborn
    missingno
    scipy
    pingouin
    scikit-posthocs
    scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import math

import matplotlib.pyplot as plt
import missingno as mno
import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
import seaborn as sns
from sklearn import linear_model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """Configuration container for file paths and core column names."""

    input_csv: str = "LevodopaChallengeWide.csv"
    separator: str = ";"
    patient_id_column: str = "patient"
    group_column: str = "Group"
    output_wide_csv: str = "output_wide.csv"


# All outcome variables analyzed in the notebook
GAIT_VARIABLES: List[str] = [
    # Spatiotemporal
    "Speed (m/s)",
    "Cadence (steps/min)",
    "Step Time - worst side (s)2",
    "Step Length - worst side (m)2",
    "Stride Time (s)",
    "Stride Length (m)",
    "Step Width (m)",
    "Stance Time - worst time t (s)2",
    "Swing Time - worst side (s)2",
    "Double Support Time (s)",
    "Single Support Time - worst side",
    # ROM
    "hip_flexion_rom_worstside",
    "hip_adduction_rom_worstside",
    "hip_rotation_rom_worstside",
    "knee_angle_rom_worstside",
    "ankle_angle_rom_worstside",
    # Mean velocity
    "hip_flexion_mean_vel_worstside",
    "hip_adduction_mean_vel_worstside",
    "hip_rotation_mean_vel_worstside",
    "knee_angle_r_mean_vel_worstside",
    "ankle_angle_r_mean_vel_worstside",
    # Variability
    "CV Stride Time",
    "Cv Stride Lenght",
    "CV Double Support",
    # Asymmetry
    "Step lenght AsymetryN",
    "Stance Time Asymetry N",
    "Swing Time Asymetry N",
]


# ---------------------------------------------------------------------------
# Missing-data imputation
# ---------------------------------------------------------------------------

def random_imputation(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Fill missing values in `feature` by sampling with replacement
    from the empirical distribution of observed values.

    A new column `<feature>_imp` must exist before calling this
    function; only its missing values are filled. [file:1]
    """
    number_missing = df[feature].isnull().sum()
    if number_missing == 0:
        return df

    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + "_imp"] = np.random.choice(
        observed_values, number_missing, replace=True
    )
    return df


def deterministic_regression_imputation(
    df: pd.DataFrame, missing_columns: Iterable[str]
) -> pd.DataFrame:
    """
    Perform deterministic regression-based imputation for a set of
    columns with missing values.

    Random imputation should be applied beforehand so that the
    auxiliary `<feature>_imp` columns exist and contain no missing
    values. [file:1]
    """
    missing_columns = list(missing_columns)
    deter_data = pd.DataFrame(columns=["Det" + name for name in missing_columns])

    for feature in missing_columns:
        # Start from the random-imputed column as baseline
        deter_data["Det" + feature] = df[feature + "_imp"]

        # Use all variables except the ones being imputed and the
        # current `_imp` column as predictors.
        predictors = list(set(df.columns) - set(missing_columns) - {feature + "_imp"})

        model = linear_model.LinearRegression()
        model.fit(X=df[predictors], y=df[feature + "_imp"])

        # Replace only where the original feature was missing
        mask_missing = df[feature].isnull()
        deter_data.loc[mask_missing, "Det" + feature] = model.predict(df[predictors])[
            mask_missing
        ]

    return deter_data


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def _setup_violin_swarm_axes(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ylabel: str,
    x_tick_labels: List[str],
    swarm_size: int = 8,
) -> None:
    """
    Create a violin + swarm plot with common styling for a given outcome
    variable across groups. [file:1]
    """
    sns.set(style="white")
    sns.set(rc={"figure.figsize": (8, 6)})

    ax = sns.violinplot(
        x=x_col,
        y=y_col,
        data=df,
        color="cadetblue",
        linewidth=0,
        alpha=1,
        scale="width",
        bw=0.2,
        cut=2,
    )
    sns.swarmplot(
        x=x_col,
        y=y_col,
        data=df,
        color="darkslategrey",
        edgecolor="darkslategrey",
        size=swarm_size,
        alpha=1,
    )

    ax.set(xlabel=None)
    ax.set(ylabel=ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    ax.set_xticklabels(x_tick_labels)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()


def plot_missing_data_matrix(df: pd.DataFrame, figsize=(24, 12)) -> None:
    """Visualize the missing-data pattern for a DataFrame. [file:1]"""
    mno.matrix(df, figsize=figsize)
    plt.tight_layout()


def plot_variable_by_group(
    df: pd.DataFrame,
    variable: str,
    group_col: str,
    ylabel: str,
    time_labels: List[str],
    swarm_size: int = 8,
) -> None:
    """
    High-level wrapper to generate violin + swarm plot for a single
    outcome variable. [file:1]
    """
    _setup_violin_swarm_axes(
        df=df,
        x_col=group_col,
        y_col=variable,
        ylabel=ylabel,
        x_tick_labels=time_labels,
        swarm_size=swarm_size,
    )


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def friedman_and_posthoc(
    df: pd.DataFrame,
    dv: str,
    within: str,
    subject: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a Friedman test and Conover post-hoc test for a given outcome
    variable and return both result tables. [file:1]
    """
    friedman_res = pg.friedman(data=df, dv=dv, within=within, subject=subject)
    posthoc_res = sp.posthoc_conover_friedman(
        a=df,
        y_col=dv,
        group_col=within,
        block_col=subject,
        p_adjust="fdr_bh",
        melted=True,
    )
    return friedman_res, posthoc_res


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def load_and_prepare_data(config: AnalysisConfig) -> pd.DataFrame:
    """
    Load the LevodopaChallengeWide dataset, drop non-numeric identifier
    columns, and perform random + regression-based imputation. [file:1]
    """
    # Load raw data
    df = pd.read_csv(config.input_csv, sep=config.separator)

    # Preserve patient identifier(s), select numeric variables only
    patient_id = df.iloc[:, 0:1]
    numeric_df = df.iloc[:, 1:282].copy()

    # Visualize missing data before imputation (optional)
    # plot_missing_data_matrix(numeric_df)

    # Identify columns with missing data (in the original wide view this
    # was simply 'list(LevodopaChallengeWide)')
    missing_columns = list(numeric_df.columns)

    # Create `_imp` columns and apply random imputation
    for feature in missing_columns:
        numeric_df[feature + "_imp"] = numeric_df[feature]
        numeric_df = random_imputation(numeric_df, feature)

    # Deterministic regression-based imputation
    deter_data = deterministic_regression_imputation(
        df=numeric_df, missing_columns=missing_columns
    )

    # Optionally visualize imputed data
    # plot_missing_data_matrix(deter_data)

    # Export imputed wide dataset
    deter_data.to_csv(config.output_wide_csv, encoding="utf-8-sig", index=False)

    # Load back with original separator so that downstream code can
    # reuse the same pipeline as in the notebook. [file:1]
    output_wide = pd.read_csv(config.output_wide_csv, sep=config.separator)

    # Reattach patient identifiers if needed
    if config.patient_id_column not in output_wide.columns and patient_id is not None:
        output_wide.insert(0, config.patient_id_column, patient_id.iloc[:, 0].values)

    return output_wide


def run_full_analysis(config: AnalysisConfig | None = None) -> None:
    """
    Execute the full pipeline:

    1. Load and impute data.
    2. Print grouped descriptive statistics for the configured outcome
       variables.
    3. Generate violin + swarm plots.
    4. Run Friedman and Conover-posthoc tests for each variable.

    This function is intended as a high-level entry point and will
    display plots and print results to stdout. [file:1]
    """
    if config is None:
        config = AnalysisConfig()

    # Load and impute data
    output_wide = load_and_prepare_data(config)

    time_labels = ["OFF", "20 min", "40 min", "60 min", "80 min"]

    for variable in GAIT_VARIABLES:
        if variable not in output_wide.columns:
            continue

        # Descriptive statistics by group
        desc = output_wide.groupby(config.group_column).describe()[variable]
        print(f"\n=== Descriptive statistics for {variable} ===")
        print(desc)

        # Plot
        ylabel = f"\n{variable}"
        plot_variable_by_group(
            df=output_wide,
            variable=variable,
            group_col=config.group_column,
            ylabel=ylabel,
            time_labels=time_labels,
        )
        plt.show()

        # Friedman + post-hoc
        if config.patient_id_column in output_wide.columns:
            friedman_res, posthoc_res = friedman_and_posthoc(
                df=output_wide,
                dv=variable,
                within=config.group_column,
                subject=config.patient_id_column,
            )
            print(f"\nFriedman test for {variable}")
            print(friedman_res)

            print(f"\nConover post-hoc (FDR-BH) for {variable}")
            print(posthoc_res)


if __name__ == "__main__":
    run_full_analysis()
