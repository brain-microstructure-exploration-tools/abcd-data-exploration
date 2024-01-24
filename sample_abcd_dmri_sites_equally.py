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

# %%
import pandas as pd
import numpy as np

# %%
# Edit these appropriately

fmriresults01_path = "./fmriresults01.txt"
abcd_y_lt_path = "./abcd_y_lt.csv"

# %%
df = pd.read_csv(fmriresults01_path, delimiter='\t', low_memory=False, dtype={'interview_age':int}, skiprows=[1])
dti_mask = df.scan_type.str.contains('dti', case=False)
dmri_subjects = df[dti_mask].src_subject_id.unique()
df = df[dti_mask]

lt = pd.read_csv(abcd_y_lt_path)
lt = lt[lt.src_subject_id.isin(dmri_subjects)]
lt.interview_age = lt.interview_age.astype(int)

df = df.merge(
    lt[['src_subject_id', 'interview_age', 'site_id_l']], 
    on=['src_subject_id', 'interview_age'], 
    how='left'
)

# %%
# For each subject we pick one of their DMRI images at random.
# Then from there for each site we pick number_to_sample_from_each_site images.
# When a site doesn't have that many images we just take all the images that it does have.

number_to_sample_from_each_site = 15

df_one_row_per_subject = df.groupby('src_subject_id').sample(1)

site_freqs = df_one_row_per_subject.site_id_l.value_counts()
sites_to_take_all_from = site_freqs[site_freqs<number_to_sample_from_each_site].index.tolist()

data_from_sites_we_took_all_from = df_one_row_per_subject[df_one_row_per_subject.site_id_l.isin(sites_to_take_all_from)]

data_from_sites_we_sampled_from = df_one_row_per_subject[~(df_one_row_per_subject.site_id_l.isin(sites_to_take_all_from))].groupby('site_id_l').sample(number_to_sample_from_each_site)

df_sample = pd.concat([data_from_sites_we_took_all_from,data_from_sites_we_sampled_from], axis=0)

# %%
# Save S3 links list for use with nda-tools downloader
with open('sample_s3_links.txt', 'w') as f:
    print('\n'.join(df_sample.derived_files), file=f)

# Save table that maps filename to site id so we can group images by site if needed later
df_sample['filename'] = df_sample.derived_files.apply(lambda s : s.split('/')[-1])
df_sample[['filename', 'site_id_l', 'fmriresults01_id']].to_csv('sample_site_table.csv')
