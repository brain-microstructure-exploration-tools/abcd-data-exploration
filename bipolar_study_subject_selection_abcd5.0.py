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
# The purpose of this notebook is to select a reasonable collection of subjects for a dimensional neuroimaging study of the onset of BP symptoms.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %% [markdown]
# The tabular ABCD data is obtained by logging in and using the download link at the [this NDA study page](https://nda.nih.gov/study.html?id=2147). Extract the zip file.

# %%
data_basedir = Path("/home/ebrahim/data/abcd/abcd-5.0-tabular-data-extracted")

# %% [markdown]
# For the 5.0 release, the data dictionary has moved to [this nifty web explorer](https://data-dict.abcdstudy.org/). For the purpose of my explorations I found it easiest to select nothing in the graph (in order to apply no filters) then select the checkbox to check *all* variables on the right. Then use the tool to download a csv data dictionary for all the variables:

# %%
data_dictionary_path = Path("/home/ebrahim/data/abcd/abcd-5.0-data-dictionary.csv")
data_dictionary = pd.read_csv(data_dictionary_path, index_col=['table_name','var_name'])

# %% [markdown]
# I think that the imaging metadata table `fmriresults01.txt`  is still downloaded the way as with ABCD 4.0 (creating a data package and using the NDA download manager or a similar tool). It's mentioned [here](https://wiki.abcdstudy.org/release-notes/imaging/overview.html) in the 5.0 release notes, but no file by the name `fmriresults01.txt` is to be found in the extracted tabular data csv files.

# %%
dmri_info_table_path = Path("/home/ebrahim/data/abcd/Package_1217694/fmriresults01.txt")

# %% [markdown]
# # Load tables

# %%
# Demographic data
demo = pd.read_csv(data_basedir/"core/abcd-general/abcd_p_demo.csv")
ldemo = pd.read_csv(data_basedir/"core/abcd-general/abcd_y_lt.csv",index_col=['src_subject_id','eventname'],parse_dates=['interview_date'])

# raw survey data for bipolar
bp= pd.read_csv(data_basedir/"core/mental-health/mh_p_ksads_bp.csv", low_memory=False) # parent survey
bp_youth = pd.read_csv(data_basedir/"core/mental-health/mh_y_ksads_bip.csv", low_memory=False) # youth survey

# mental health data
mh = pd.read_csv(data_basedir/"core/mental-health/mh_y_ksads_ss.csv")

# DMRI imaging data
dmri = pd.read_csv(dmri_info_table_path,sep='\t',header=0,skiprows=[1],parse_dates=['interview_date'])

# %% [markdown]
# In the 4.0 notebook we removed a couple of subject keys from the tables because their 2 and 3 year follow ups occured around the same interview age. This may have been an error that was corrected in 5.0 because it's no longer the case when we look in the table `ldemo`.

# %% [markdown]
# # Subject Selection

# %% [markdown]
# There are up to 3 time points for these interviews:

# %%
bp.eventname.value_counts()

# %% [markdown]
# Look at counts for the 1-year follow-up data... it looks like release 5.0 dropped a lot of the 1-year follow-up data that was in 4.0! That's for the raw bipolar interview questions. Something must have gone wrong in the 5.0 release of the raw KSADS interview questions here. Same issue in the youth survey:

# %%
bp_youth.eventname.value_counts()

# %% [markdown]
# But the general diagnostics table appears to be okay still:

# %%
mh.eventname.value_counts()

# %% [markdown]
# In fact we see here that there are up to *five* time points. Somehow we have raw answers to KSADS questions in for only baseline and 2-year follow-up (and in rare cases also 1-year follow-up), while we have the KSADS symptoms and diagnostics across all five time points.
#
# I'll use the KSADS diagnostics table `mh` instead of the KSADS raw answers tables `bp,bp_youth` to do subject selection this time, since something is wrong with the raw answers tables.

# %% [markdown]
# Our largest number of subjects to select is the number who have mental health info from at least one time point:

# %%
mh.src_subject_id.nunique()

# %% [markdown]
# Or we could only keep the subjects who did at least 4 of the mental health interviews:

# %%
sum(mh.groupby('src_subject_id').apply(lambda x : len(x)) >= 4)

# %% [markdown]
# We can instead focus on subjects who had any BP symptom as indicated by KSADS diagnistics, with a sampling of the same number of healthy controls. Let's look what are the bipolar-related columns in the KSADS diagnostics table:

# %%
mh_dd = data_dictionary.loc['mh_y_ksads_ss']
for i,r in mh_dd[mh_dd.var_label.str.contains("Bipolar", case=False)].iterrows():
    print(f"{i}: {r.var_label}")

# %% [markdown]
# Hand-pick based on whether they seem to indicate that some sort of BP may be present:

# %%
bp_elementnames = [
    'ksads_2_207_t', # Symptom - Psychomotor Agitation in Bipolar Disorder, Present
    'ksads_2_208_t', # Symptom - Psychomotor Agitation in Bipolar Disorder, Past
    'ksads_2_215_t', # Symptom - Impairment in functioning due to bipolar, Present
    'ksads_2_216_t', # Symptom - Impairment in functioning due to bipolar, Past
    'ksads_2_217_t', # Symptom - Hospitalized due to Bipolar Disorder, Present
    'ksads_2_218_t', # Symptom - Hospitalized due to Bipolar Disorder, Past
    'ksads_2_830_t', # Diagnosis - Bipolar I Disorder, current episode manic (F31.1x)
    'ksads_2_831_t', # Diagnosis - Bipolar I Disorder, current episode depressed, F31.3x
    'ksads_2_832_t', # Diagnosis - Bipolar I Disorder, currently hypomanic  F31.0
    'ksads_2_833_t', # Diagnosis - Bipolar I Disorder, most recent past episode manic (F31.1x)
    'ksads_2_834_t', # Diagnosis - Bipolar I Disorder, most recent past episode depressed (F31.1.3x)
    'ksads_2_835_t', # Diagnosis - Bipolar II Disorder, currently hypomanic F31.81
    'ksads_2_836_t', # Diagnosis - Bipolar II Disorder, currently depressed F31.81
    'ksads_2_837_t', # Diagnosis - Bipolar II Disorder, most recent past hypomanic F31.81
    'ksads_2_838_t', # Diagnosis - Unspecified Bipolar and Related Disorder, current (F31.9)
    'ksads_2_839_t', # Diagnosis - Unspecified Bipolar and Related Disorder, PAST (F31.9)
    'ksads2_2_803_t', # Diagnosis - Other Specified Bipolar and Related Disorder (F31.9) Meets criteria for Bipolar II disorder except duration criteria
    'ksads2_2_931_t', # Diagnosis - Bipolar II Disorder, Currently Depressed F31.81
    'ksads2_2_932_t', # Diagnosis - Other Specified Bipolar and Related Disorder (F31.9) Meets criteria for Bipolar II disorder except never had major depressive episode
    'ksads2_2_933_t', # Diagnosis - Other Specified Bipolar and Related Disorder (F31.9) Recurrent manic or hypomanic episodes of shorter duration than minimum criteria
    'ksads2_2_936_t', # Bipolar II Disorder, Currently Depressed (in partial remission) F31.81
    'ksads2_2_937_t', # Bipolar I Disorder, Currently Depressed (in partial remission) F31.3x
    'ksads2_2_802_t', # Diagnosis - Bipolar II Disorder, most recent episode hypomanic F31.81
    'ksads2_2_193_t', # Symptom - Psychomotor Agitation in Bipolar Disorder, Present
    'ksads2_2_194_t', # Symptom - Psychomotor Agitation in Bipolar Disorder, Past
    'ksads2_2_201_t', # Symptom - Impairment in functioning due to bipolar, Present
    'ksads2_2_202_t', # Symptom - Impairment in functioning due to bipolar, Past
    'ksads2_2_203_t', # Symptom - Hospitalized due to Bipolar Disorder, Present
    'ksads2_2_204_t', # Symptom - Hospitalized due to Bipolar Disorder, Past
    'ksads2_2_798_t', # Diagnosis - Bipolar I Disorder, current episode manic (F31.1x)
    'ksads2_2_799_t', # Diagnosis - Bipolar I Disorder, current episode depressed, F31.3x
    'ksads2_2_800_t', # Diagnosis - Bipolar I Disorder, most recent episode manic (F31.9)
    'ksads2_2_801_t', # Diagnosis - Bipolar II Disorder, currently hypomanic F31.81
]

# %% [markdown]
# Hand-pick a subset based on whether they are more directly indicative of a bipolar diagnosis:

# %%
bp_elementnames2 = [
    'ksads_2_830_t', # Diagnosis - Bipolar I Disorder, current episode manic (F31.1x)
    'ksads_2_831_t', # Diagnosis - Bipolar I Disorder, current episode depressed, F31.3x
    'ksads_2_832_t', # Diagnosis - Bipolar I Disorder, currently hypomanic  F31.0
    'ksads_2_833_t', # Diagnosis - Bipolar I Disorder, most recent past episode manic (F31.1x)
    'ksads_2_834_t', # Diagnosis - Bipolar I Disorder, most recent past episode depressed (F31.1.3x)
    'ksads_2_835_t', # Diagnosis - Bipolar II Disorder, currently hypomanic F31.81
    'ksads_2_836_t', # Diagnosis - Bipolar II Disorder, currently depressed F31.81
    'ksads_2_837_t', # Diagnosis - Bipolar II Disorder, most recent past hypomanic F31.81
    'ksads2_2_931_t', # Diagnosis - Bipolar II Disorder, Currently Depressed F31.81
    'ksads2_2_936_t', # Bipolar II Disorder, Currently Depressed (in partial remission) F31.81
    'ksads2_2_937_t', # Bipolar I Disorder, Currently Depressed (in partial remission) F31.3x
    'ksads2_2_802_t', # Diagnosis - Bipolar II Disorder, most recent episode hypomanic F31.81
    'ksads2_2_798_t', # Diagnosis - Bipolar I Disorder, current episode manic (F31.1x)
    'ksads2_2_799_t', # Diagnosis - Bipolar I Disorder, current episode depressed, F31.3x
    'ksads2_2_800_t', # Diagnosis - Bipolar I Disorder, most recent episode manic (F31.9)
    'ksads2_2_801_t', # Diagnosis - Bipolar II Disorder, currently hypomanic F31.81
]

# %% [markdown]
# Here is the number of subjects that at some time had a "1" in each of the two respective sets of columns:

# %%
mh['some_bp_thing'] = mh[bp_elementnames].apply(lambda x : (x==1).any(), axis=1)
mh['some_bp_thing2'] = mh[bp_elementnames2].apply(lambda x : (x==1).any(), axis=1)
print(mh.groupby('src_subject_id').some_bp_thing.agg(func='any').sum())
print(mh.groupby('src_subject_id').some_bp_thing2.agg(func='any').sum())

# %% [markdown]
# That last one seems like a good number to take: 878 subjects. Let's now restrict to the subset that also have good longitudinal data.

# %%
# indicates whether a given *subject* had at least one interview that contained a positive diagnostic
subject_positive_bp = mh.groupby('src_subject_id').some_bp_thing2.agg(func='any')

# indicates whether a given subject did at least four of the 5 mental health interviews
subject_interview_complete = mh.groupby('src_subject_id').apply(lambda x : len(x)) >= 4

# indicates whether a given subject did at least two scans
subject_scans_complete = dmri.groupby(['subjectkey']).apply(lambda x : x.interview_age.nunique()) == 2

# %% [markdown]
# If we want to include the subjects that satisfy all three of these criteria, then here's how many subjects we get:

# %%
(subject_positive_bp & subject_interview_complete & subject_scans_complete).sum()

# %% [markdown]
# However after looking at the sorts of subjects we get from this and when the BP indicators show up... a large number seem to only have BP indicated at the baseline interview and then never again at a later interview. Let's strengthen our filter to only include those who had some BP indication beyond just the baseline interview. It just seems not believable to have the prevalence be so much higher at baseline; that was likely a case of overdiagnosing early on. We may later want to restrict the filtering even more to only include those whose BP indication comes from KSADS 2, so from the 3-year follow-up onward, since KSADS 2 is known to be better at not overdiagnosing.

# %%
subject_positive_bp_beyond_baseline = mh[mh.eventname!="baseline_year_1_arm_1"].groupby('src_subject_id').some_bp_thing2.agg(func='any')

# %%
subject_positive_bp_beyond_baseline.sum()

# %%
(subject_positive_bp_beyond_baseline & subject_interview_complete & subject_scans_complete).sum()

# %%
subject_inclusion_criteria = pd.concat(
    {'positive_bp':subject_positive_bp_beyond_baseline,
     'interview_complete':subject_interview_complete,
     'scans_complete':subject_scans_complete
    }, axis=1)

subject_inclusion_criteria = subject_inclusion_criteria.replace(np.nan, False)

# %% [markdown]
# If we then want to include the same number of healthy controls, then we can take a random sample that is stratified based on the demographics of these 190. (Or should it be stratified based on the demographics of the full abcd study cohort?)

# %% [markdown]
# # COVID-19 Lockdown Effects
#
# ## Data Collection Timing
# Let's now look at the interview dates for those included subjects.

# %%
included_subjects = subject_inclusion_criteria[subject_inclusion_criteria.all(axis=1)].index

# %%
time_index_mapping = {
    'baselineYear1Arm1' : 0,
    '2YearFollowUpYArm1' : 1,
    '4YearFollowUpYArm1' : 2,
}
dmri['time_index'] = dmri.derived_files.apply(lambda x : time_index_mapping[x.split('/')[-1].split('_')[1]])


# %%
def year_and_month_to_month_offset(year, month, day):
    return (year-2018)*12 + month + day/30.5

dmri_second_scan_included_subjects = dmri[dmri.subjectkey.isin(included_subjects) & (dmri.time_index==1)]
dmri_second_scan_included_unique_subjects = dmri_second_scan_included_subjects.groupby('subjectkey').apply(lambda x : x.iloc[0])
month_offsets = dmri_second_scan_included_unique_subjects.interview_date.apply(lambda x : year_and_month_to_month_offset(x.year,x.month, x.day))
lockdown_month_offset = year_and_month_to_month_offset(2020,3,20)

# %%
bins = range(8,38,2)
plt.figure(figsize=(4,2))
plt.hist(month_offsets, bins=bins)
plt.plot([lockdown_month_offset,lockdown_month_offset],[0,100], label="Lockdown")
plt.xticks(ticks = bins)
plt.ylim(0,25)
plt.yticks(ticks=range(0,30,10))
plt.xlabel("Months since Jan 2018")
plt.title("2-Year Follow-Up Scan Dates (ABCD 5.0)")
plt.legend()
plt.show()

# %%
(month_offsets > lockdown_month_offset).value_counts()

# %% [markdown]
# This is great. Of the 190 selected BP subjects, 61 of them had their second scan after lockdown measures, and 108 of them before. That means we can _ignore_ lockdown effects by restricting to those 108 and it's not a serious loss at all. And we can also _observe_ lockdown effects if we want to, because 61 is a decent number of positive post-lockdown cases to have.

# %% [markdown]
# Let's now check the same for the mental health interview data.

# %%
time_index_mapping = {
    'baseline_year_1_arm_1' : 0,
    '1_year_follow_up_y_arm_1' : 1,
    '2_year_follow_up_y_arm_1' : 2,
    '3_year_follow_up_y_arm_1' : 3,
    '4_year_follow_up_y_arm_1' : 4,
}
mh['time_index'] = mh.eventname.apply(lambda x : time_index_mapping[x])

def join_interview_date_from_ldemo_table(df):
    return df.join(ldemo[['interview_date']], on=['src_subject_id','eventname'])

mh_time1_included_subjects = mh[mh.src_subject_id.isin(included_subjects) & (mh.time_index==1)]
mh_time1_included_subjects = join_interview_date_from_ldemo_table(mh_time1_included_subjects)
month_offsets1 = mh_time1_included_subjects.interview_date.apply(lambda x : year_and_month_to_month_offset(x.year,x.month, x.day))
mh_time2_included_subjects = mh[mh.src_subject_id.isin(included_subjects) & (mh.time_index==2)]
mh_time2_included_subjects = join_interview_date_from_ldemo_table(mh_time2_included_subjects)
month_offsets2 = mh_time2_included_subjects.interview_date.apply(lambda x : year_and_month_to_month_offset(x.year,x.month, x.day))

# %%
bins = range(8,38,2)
plt.figure(figsize=(4,2))
plt.hist(month_offsets2, bins=bins)
plt.plot([lockdown_month_offset,lockdown_month_offset],[0,100], label="Lockdown")
plt.xticks(ticks = bins)
plt.ylim(0,25)
plt.yticks(ticks=range(0,30,10))
plt.xlabel("Months since Jan 2018")
plt.title("2-Year Follow-Up Interview Dates (ABCD 5.0)")
plt.legend()
plt.show()
