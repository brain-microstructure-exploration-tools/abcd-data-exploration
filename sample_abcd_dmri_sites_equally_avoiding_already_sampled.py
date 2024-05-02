# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# The purpose of this script is to sample the ABCD Study diffusion MRI images such that each subject is sampled at most once and each study site is sampled pretty much equally.
#
# For each subject we will pick one of their DMRI images at random.
# Then from there for each study site we pick a certain fixed number of images
# (or all images from that site if it doesn't have at least that number).
#
# To do this we will need the tables `fmriresults01.txt` and `abcd_y_lt.csv` from [the ABCD study](https://wiki.abcdstudy.org/release-notes/start-page.html). This script is meant for ABCD Study release 5.0.
#
# **Importance difference between this and `sample_abcd_dmri_sites_equally.py`:** This script was created to run a similar sampling procudre while also avoiding items that have already been sampled. The purpose is to create a dataset on which to evaluate algorithms that were generated using a previous sampling.

# %%
import pandas as pd
import numpy as np
from pathlib import Path

# %%
# Edit these appropriately

data_root_path = Path("/data/ebrahim-data/abcd")
fmriresults01_path = data_root_path/"Package_1224700/fmriresults01.txt"
abcd_y_lt_path = data_root_path/"abcd-5.0-tabular-data-extracted/core/abcd-general/abcd_y_lt.csv"
mri_y_adm_info_path = data_root_path/"abcd-5.0-tabular-data-extracted/core/imaging/mri_y_adm_info.csv"
previous_sample_path = Path.home()/'Desktop/previous_sample/sample_site_table.csv'

# %%
df = pd.read_csv(fmriresults01_path, delimiter='\t', low_memory=False, dtype={'interview_age':int}, skiprows=[1])
dti_mask = df.scan_type.str.contains('dti', case=False)
dmri_subjects = df[dti_mask].src_subject_id.unique()
df = df[dti_mask]

lt = pd.read_csv(abcd_y_lt_path)
lt = lt[lt.src_subject_id.isin(dmri_subjects)]
lt.interview_age = lt.interview_age.astype(int)

mri_info = pd.read_csv(mri_y_adm_info_path)

df = df.merge(
    lt[['src_subject_id', 'interview_age', 'site_id_l', 'eventname']], 
    on=['src_subject_id', 'interview_age'], 
    how='left'
)

df = df.merge(
    mri_info[['src_subject_id', 'eventname', 'mri_info_manufacturer', 'mri_info_manufacturersmn']], 
    on=['src_subject_id', 'eventname'], 
    how='left'
)

df = df.dropna(subset = ['mri_info_manufacturer'])

df = df.reset_index(drop=True)

# %%
# Drop the rows that were previously sampled

previous_sample_df = pd.read_csv(previous_sample_path)
df = df[~(df.fmriresults01_id.isin(previous_sample_df.fmriresults01_id))]

# %%
# For each subject we pick one of their DMRI images (i.e. time points) at random.

subject_to_selected_timepoint = df.groupby('src_subject_id').apply(lambda g : np.random.choice(g.eventname.unique()))

subject_to_selected_timepoint.name = 'selected_eventname'
df_one_time_per_subject = df.merge(
    subject_to_selected_timepoint,
    on="src_subject_id",
)
df_one_time_per_subject = df_one_time_per_subject[df_one_time_per_subject.eventname == df_one_time_per_subject.selected_eventname]

# %%
# Then from there for each site we pick number_to_sample_from_each_site images.
# When a site doesn't have that many images we just take all the images that it does have.
# (Care is taken ensure that the sample is not biased by the fact that the phillips scans are split
# over multiple files and hence have multiple rows per scan. This is why it's so messy.)

number_to_sample_from_each_site = 4

site_freqs = df_one_time_per_subject.groupby('src_subject_id').apply(lambda g : g.iloc[0]).site_id_l.value_counts()
sites_to_take_all_from = site_freqs[site_freqs<number_to_sample_from_each_site].index.tolist()

data_from_sites_we_took_all_from = df_one_time_per_subject[df_one_time_per_subject.site_id_l.isin(sites_to_take_all_from)]

subjects_from_sites_we_sampled_from = df_one_time_per_subject[~(df_one_time_per_subject.site_id_l.isin(sites_to_take_all_from))].groupby('src_subject_id').apply(lambda g : g.iloc[0]).groupby('site_id_l').sample(number_to_sample_from_each_site).index
data_from_sites_we_sampled_from = df_one_time_per_subject[df_one_time_per_subject.src_subject_id.isin(subjects_from_sites_we_sampled_from)]

df_sample = pd.concat([data_from_sites_we_took_all_from,data_from_sites_we_sampled_from], axis=0)

# %%
# Save S3 links list for use with nda-tools downloader
with open('sample_s3_links.txt', 'w') as f:
    print('\n'.join(df_sample.derived_files), file=f)

# Save table that maps filename to site id so we can group images by site if needed later
df_sample['filename'] = df_sample.derived_files.apply(lambda s : s.split('/')[-1])
df_sample[['filename', 'site_id_l', 'mri_info_manufacturer', 'fmriresults01_id',]].to_csv('sample_site_table.csv', index=False)
