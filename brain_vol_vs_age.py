# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# The purpose of this notebook is to plot overall cortical volume variation and trends over age in order to determine whether scaling rather than merely rigid registration is needed when building population templates.

# %%
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path

# %% [markdown]
# The tabular ABCD data is obtained by logging in and using the download link at the [this NDA study page](https://nda.nih.gov/study.html?id=2147). Extract the zip file.

# %%
data_basedir = Path("/home/ebrahim/data/abcd/abcd-5.0-tabular-data-extracted")

# %% [markdown]
# For the 5.0 release, the data dictionary can be found at [this web explorer](https://data-dict.abcdstudy.org/). For the purpose of my explorations I found it easiest to select nothing in the graph (in order to apply no filters) then select the checkbox to check *all* variables on the right. Then use the tool to download a csv data dictionary for all the variables:

# %%
data_dictionary_path = Path("/home/ebrahim/data/abcd/abcd-5.0-data-dictionary.csv")
data_dictionary = pd.read_csv(data_dictionary_path, index_col=['table_name','var_name'])

# %% [markdown]
# # Load tables

# %%
# Demographic data
demo = pd.read_csv(data_basedir/"core/abcd-general/abcd_p_demo.csv")
ldemo = pd.read_csv(data_basedir/"core/abcd-general/abcd_y_lt.csv",parse_dates=['interview_date'])

# sMRI - Volume (Desikan) table
mri_y_smr_vol_dsk = pd.read_csv(data_basedir/"core/imaging/mri_y_smr_vol_dsk.csv", low_memory=False)

# Quality Control - Raw - Structural MRI - T1
mri_y_qc_raw_smr_t1 = pd.read_csv(data_basedir/"core/imaging/mri_y_qc_raw_smr_t1.csv", low_memory=False)

# Quality Control - Raw - Diffusion MRI
mri_y_qc_raw_dmr = pd.read_csv(data_basedir/"core/imaging/mri_y_qc_raw_dmr.csv", low_memory=False)


# %% [markdown]
# # Plot and get correlation

# %%
def plot_and_print_correlation(df, ycol):
    """ Given an abcd data table and a numerical column y, make a scatterplot of y against age,
    and print the correlation of y to age.
    """    
    d = df.merge(ldemo, on=['src_subject_id', 'eventname'])
    xcol = 'interview_age'
    d = d.dropna(subset=[xcol,ycol])
    x = d[xcol]
    y = d[ycol] # T1: Mean volume of brain mask averaged across all scans (mm^3)
    plt.scatter(x,y, marker='.')
    plt.show()
    corr, p_value = pearsonr(x, y)
    print(f"corr: {corr}, p-val: {p_value}")


# %%
plot_and_print_correlation(mri_y_smr_vol_dsk, 'smri_vol_cdk_total')

# %%
plot_and_print_correlation(df = mri_y_qc_raw_smr_t1, ycol = 'iqc_t1_all_brainvol')

# %%
plot_and_print_correlation(df = mri_y_qc_raw_dmr, ycol = 'iqc_dmri_ok_brainvol')

# %% [markdown]
# Hmm quite mixed results depending on the notion of brain volume used.
