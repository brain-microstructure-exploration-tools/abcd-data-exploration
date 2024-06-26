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
# This script processes the output of `sample_abcd_dmri_sites_equally.py` to grab the imaging data s3 links for other modalities besides just DMRI.
#
# We need the tables `fmriresults01.txt`, `abcd_y_lt.csv`, and `mri_y_adm_info.csv` from [the ABCD study](https://wiki.abcdstudy.org/release-notes/start-page.html). We need the output `sample_site_table.csv` from `sample_abcd_dmri_sites_equally.py`. This script is meant for ABCD Study release 5.0 or 5.1. In this script I use two versions of `fmriresults01.txt`, since I created a separate data package that has the t1 and t2 but not the dmri.

# %%
import pandas as pd
import numpy as np
from pathlib import Path

# %%
# Edit these appropriately

data_root_path = Path("/data/ebrahim-data/abcd")
fmriresults01_dmri_path = data_root_path/"Package_1224700/fmriresults01.txt"
fmriresults01_t1t2_path = data_root_path/"Package_1227348/fmriresults01.txt"
abcd_y_lt_path = data_root_path/"abcd-5.0-tabular-data-extracted/core/abcd-general/abcd_y_lt.csv"
mri_y_adm_info_path = data_root_path/"abcd-5.0-tabular-data-extracted/core/imaging/mri_y_adm_info.csv"
sample_site_table_path = Path("./sample_site_table.csv")


# %%
def read_fmriresults_file_and_merge_other_tables(fmriresults01_path):
    df = pd.read_csv(fmriresults01_path, delimiter='\t', low_memory=False, dtype={'interview_age':int}, skiprows=[1])
    
    lt = pd.read_csv(abcd_y_lt_path)
    lt = lt.dropna(axis=0,subset=['interview_age'])
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
    return df


# %%
df_dmri = read_fmriresults_file_and_merge_other_tables(fmriresults01_dmri_path)
df_t1t2 = read_fmriresults_file_and_merge_other_tables(fmriresults01_t1t2_path)

# %%
fmriresults01_ids = pd.read_csv(sample_site_table_path).fmriresults01_id

df_rows_from_original_dmri_sample = df_dmri[df_dmri.fmriresults01_id.isin(fmriresults01_ids)]
desired_subject_and_event_series = pd.Series(zip(df_rows_from_original_dmri_sample.src_subject_id, df_rows_from_original_dmri_sample.eventname))

def get_scan_type_for_sampled_interviews(df, scan_type):
    df_subject_and_event_series = pd.Series(zip(df.src_subject_id, df.eventname))
    return df[df_subject_and_event_series.isin(desired_subject_and_event_series) & df.scan_type.str.contains(scan_type,case=False)].copy()

df_sample = pd.concat(
    [
        get_scan_type_for_sampled_interviews(df_t1t2, 't1'),
        get_scan_type_for_sampled_interviews(df_t1t2, 't2'),
    ],
    axis=0,
)

# %%
# Save S3 links list for use with nda-tools downloader
with open('sample_s3_links_t1t2.txt', 'w') as f:
    print('\n'.join(df_sample.derived_files), file=f)

# Save table that maps filename to site id so we can group images by site if needed later
df_sample['filename'] = df_sample.derived_files.apply(lambda s : s.split('/')[-1])
df_sample[['filename', 'site_id_l', 'mri_info_manufacturer', 'fmriresults01_id',]].to_csv('sample_site_table_t1t2.csv', index=False)
