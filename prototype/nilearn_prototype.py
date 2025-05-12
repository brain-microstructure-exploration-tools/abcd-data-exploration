# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: abcd311
#     language: python
#     name: abcd311
# ---

# # An example workflow for generating hypotheses with ABCD data
#
# ## Overview
# Big picture, this workbook shows how to use [Adolescent Brain Cognitive Development Study (ABCD
# Study®)](https://abcdstudy.org/) subject data and metadata about their images to predict voxel values in the images.
# The goal is generating hypotheses about which voxels are associated with those data.
#
# ### Input & Output
# The input data are divided into three categories:
# * **tested variables:** these are variables that we are seeking significant voxels for.  These are typically
#   [KSADS](https://en.wikipedia.org/wiki/Kiddie_Schedule_for_Affective_Disorders_and_Schizophrenia) data from
#   `mental-health/mh_y_ksads_ss`.  However for illustrative purposes in this notebook, we also test for subject's
#   gender.
# * **confounding variables:** these are variables that may affect voxel values and hide the significant signal that we
#   are hoping to detect for the tested variables.  These can be variables like age
#   (`abcd-general/abcd_y_lt/interview_age`), hospital (`abcd-general/abcd_y_lt/site_id_l`), gender
#   (`gender-identity-sexual-health/gish_p_gi/demo_gender_id_v2`), or family (`abcd-general/abcd_y_lt/rel_family_id`).
#   * These variables can be used for both intercept and slope statistical random effects in a [multilevel
#     model](https://en.wikipedia.org/wiki/Multilevel_model).
# * **voxel data:** the intensity associated with each voxel of each registered image.  The data set includes both "fa"
#   and "md" images, and a separate analysis is run for each.
#
# The output includes:
# * **a computed statistical significance for each voxel for each tested variable:** This is output as one image per
#   tested variable.
# Each of these images can be visualized in software such as [3D Slicer](https://download.slicer.org/), alongside a
# registered segmentation image that labels brain regions.  Those voxels with a high statistical significance are also
# output by this notebook in text form.  Furthermore, the X-slice, Y-slice, and Z-slice with the most combined
# statistical significance are shown.
#
# ### Methods
# Mathematically speaking, for each tested variable separately, we process each voxel separately.  We use a [multlevel
# model](https://en.wikipedia.org/wiki/Multilevel_model) to predict the voxel's intensity as a function of the tested
# variable and the confounding variables.  This is a regression model with a non-zero constant term permitted.  When a
# confounding variable is an ordered variable, it is used directly.  If a confounding variable is an unordered variable,
# (such as a site id or other category indicator), it is represented in [one-hot](https://en.wikipedia.org/wiki/One-hot)
# notation; with a separate [indicator term](https://en.wikipedia.org/wiki/Indicator_function) for all categories but
# one.  When a confounding variable is labeled as "intercept" these values are used directly as just described.  When
# one confounding variable is labeled as "time" each confounding variable that is labeled as "slope" adds a term to the
# regression that is the product of the time variable and the "slope" variable.  (A given confounding variable can be
# labeled as more than one of "time", "intercept", and "slope", though only one confounding variable can be labeled as
# "time".)  Together these intercept and slope variables are often called [random
# effects](https://en.wikipedia.org/wiki/Random_effects_model).
#
# In equations:
# $$v_i = \sum_c \beta_c D_{ci} + \epsilon_i$$
# where
# * $v_i$ is the intensity of the voxel in image $i$
# * $c$ indexes the independent variables of the regression, which are the form of any of:
#   * an ordered confounding variable
#   * a category of an unordered confounding variable
#   * one of the above two possibilities multiplied by a time factor (so as to make a random effects slope)
#   * the tested variable
#   * a constant term
# * $\beta_c$ is the regression coefficient for term $c$, which we are solving for
# * $D_{ci}$ is the input-data value of the $c$ term for image $i$.
# * $\epsilon_i$ is the error term for image $i$.  The regression minimizes the sum of squares of these values.
#
# The regression is solved and the return value is
# $\log_{10}(1 / \operatorname{pvalue}(\operatorname{tstatistic}(\beta_t)))$, where $\beta_t$ is the coefficient for the
# tested variable and the t-statistic is quantifying the belief that $\beta_t \ne 0$.  Furthermore, nilearn adjusts for
# multiple testing.  A value of $2.0$ indicates a p-value of 1%, a value of $1.3$ indicates a p-value of 5%, a value of
# $1.0$ indicates a p-value of 10%, and so on.
#
# ## Individual steps
# In addition to showing a high-level approach to hypothesis generation, this is functioning code that also provides
# many low-level details.
#
# ### Installing dependencies
# This is Python code that relies on some well-established, useful packages available for Python.  This workbook shows
# how to get those packages.
#
# ### Functions
# Various steps are shown in the functions that are defined below.  Ultimately, these functions are invoked towards the
# end of this notebook.
# * **get_list_of_image_files:** demonstrates how to select only voxel-data files from a given directory.
# * **parse_image_filenames:** demonstrates how to extract image metadata from each image file's name and load it into a
#   dataframe.
# * **get_data_from_image_files:** demonstrates how to read voxel data from each image file.
# * **get_white_matter_mask_as_numpy:** ** demonstrates how to create a mask that indicates which voxels are white
#   matter.
# * **csv_file_to_dataframe:** demonstrates how to read a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values)
#   file into a Python [pandas](https://pandas.pydata.org/) dataframe.
# * **select_rows_of_dataframe:** is a deprecated function replaced by pandas' `isin` functionality.
# * **clean_ksads_data_frame:** demonstrates how missing values in KSADS data can be handled.
# * **ksads_filename_to_dataframe:** demonstrates how to load and preprocess the KSADS data.
# * **data_frame_value_counts:** demonstrates how to do a census of answers separately for each column of KSADS data, as
#   part of the process of picking which KSADS variables are likely interesting.
# * **entropy_of_column_counts:** demonstrates how to compute the entropy/information of one KSADS variable from its
#   census of answers, as part of the process of picking which KSADS variables are likely interesting.
# * **ksads_keys_only:** demonstrates how to filter out non-KSADS columns from a pandas dataframe.
# * **entropy_of_all_columns:** demonstrates how to apply entropy_of_column_counts to all KSADS variables, one at a
#   time.
# * **find_interesting_entropies:** demonstrates how to call data_frame_value_counts and entropy_of_all_columns to
#   calculate and support the KSADS variables' entropy/information values.
# * **find_interesting_ksads:** demonstrates how a user can select KSADS variables for further analysis based upon how
#   much entropy/information is in the dataset for each KSADS variable.
# * **large_enough_category_size:** demonstrates how to remove un- or barely informative confounding variables.
# * **process_confounding_var:** demonstrates how to collect and do most of the processing for each confounding
#   variable.
# * **process_longitudinal_config:** demonstrates how confounding variables can be processed as intercept or slope
#   random effects.
# * **make_dataframe_for_confounding_vars:** demonstrates how to call subroutines to retrieve, process, and combine all
#   confounding variables into a single dataframe.
# * **merge_confounding_table:** demonstrates how to combine confounding variables data with associated image metadata.
# * **find_good_slice:** demonstrates how to find an interesting X-, Y-, or Z-slice in the output data.
# * **find_local_maxima:** demonstrates how to find voxels that are local maxima in terms of statistical significance.
# * **use_nilearn:** demonstrates how to read and preprocess all the input, use the `nilearn` library for processing,
#   and postprocess the output.
#
# ### Running
# The top-level routine that is invoked is `use_nilearn`.

# ## Installing dependent Python packages
# One way to do this is with a virtual python environment installed so that it is accessible to Jupyter lab.  This needs
# to be set up only once.  I am using Python 3.11.8, but this likely works with other versions of Python.
# ```bash
# python -m venv ~/abcd311
# source ~/abcd311/bin/activate
# pip install -U ipykernel jupyterlab jupytext pip setuptools wheel
# python -m ipykernel install --user --name abcd311
# # cd abcd-data-exploration
# pip install -r requirements.txt
# ```
# Then, once Jupyter is open with this lab notebook, "Change Kernel..." to be "abcd311".

# ## Creating a Jupyter lab notebook from a Python file
# Use a command like
# ```bash
# jupytext --to ipynb nilearn_prototype.py
# ```
# to create `nilearn_prototype.ipynb`, which can be run with something like `jupyter lab`.

# ## Import dependent Python packages

# +
# Import from Python packages
from __future__ import annotations

import functools
import math
import os
import re
import sys
import time
from re import Match
from typing import Any, cast

import dipy.io.image
import matplotlib.pyplot
import nibabel
import nibabel.nifti1
import nilearn
import nilearn.maskers
import nilearn.masking
import nilearn.mass_univariate
import numpy as np
import pandas as pd
import scipy.signal
import slicerio

# -

# ## Set global variables
# ### Inputs

# Set global parameters to match your environment.  Ultimately these will be member variables of a class.
gor_image_directory: str = "/data2/ABCD/gor-images"
# TODO: We may need separate compute_white_matter_mask_from_file files for "fa" and "md"
compute_white_matter_mask_from_file: str = os.path.join(gor_image_directory, "gortemplate0.nii.gz")
coregistered_images_directory: str = os.path.join(gor_image_directory, "coregistered-images")
tabular_data_directory: str = "/data2/ABCD/abcd-5.0-tabular-data-extracted"
core_directory: str = os.path.join(tabular_data_directory, "core")
# fa_seg_ files are originally from /data/lee-data/abcd/registration-experiments/2024-01
fa_seg_directory: str = "/home/local/KHQ/lee.newberg/git/brain-microstructure-exploration-tools/abcd-data-exploration"
fa_partition_filename: str = os.path.join(fa_seg_directory, "prototype/segmentation_labelmap_mrtrix.seg.nrrd")
fa_cloud_filename: str = os.path.join(fa_seg_directory, "prototype/segmentation_data_mrtrix.seg.nrrd")

# ### Outputs

# Set global parameters to match your environment.  Ultimately these will be member variables of a class.
an_output_filename: str = "/home/local/KHQ/lee.newberg/demo_gender_id_v2==1.0.nii.gz"

# ### Internal configuration

# +
# Ultimately these will be static const members a class

# Images are uniquely identified / distinguished from each other by their subjects and timing.
# (They are also distinguished as "md" vs. "fa" image_subtype, though not relevant here.)
join_keys_global: list[str] = ["src_subject_id", "eventname"]
# Defaults for confounding_vars_config
LongitudinalDefault: list[str] = ["intercept"]
IsMissingDefault: list[Any] = ["", np.nan]
ConvertDefault: dict[Any, Any] = {}
# TODO: Make sure that nilearn.masking.compute_brain_mask() is using `threshold` the way we think it does
mask_threshold_global: float = 0.70
min_category_size_global: int = 5
background_index: int = 0
# -

# ## Define functions for various steps of the workflow

# +
# Functions for handling image voxel data


def get_list_of_image_files(directory: str) -> list[str]:
    """
    Returns a list of full path names of image files.  Input is the directory in which to look for
    these files.
    """
    # Choose the pattern to get .nii.gz files but avoid the template files such as gortemplate0.nii.gz
    pattern: str = r"^gorinput[0-9]{4}-.*\.nii\.gz$"
    response: list[str] = [
        os.path.join(directory, file) for file in os.listdir(directory) if bool(re.match(pattern, file))
    ]
    return response


def parse_image_filenames(list_of_image_files: list[str]) -> pd.core.frame.DataFrame:
    """
    Given a list of image file names, returns a pandas DataFrame.
    The first column in the output is the filename.
    Additional columns indicate how the filename was parsed.
    For example, run as:
        df = parse_image_filenames(get_list_of_image_files(coregistered_images_directory))
    """
    filename_pattern: str = (
        r"gorinput([0-9]+)-modality([0-9]+)-sub-([A-Za-z0-9]+)_ses-([A-Za-z0-9]+)_run-"
        r"([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)-([A-Za-z0-9]+).nii.gz"
    )
    filename_keys: list[str] = [
        "filename",
        "gorinput",
        "modality",
        "src_subject_id",
        "eventname",
        "run",
        "image_type",
        "image_subtype",
        "processing",
    ]

    # Parse the basename of each filename and use it to construct a row of the `response` dataframe
    response: pd.core.frame.DataFrame = pd.DataFrame(
        [
            [filename, *list(cast(Match[str], re.match(filename_pattern, os.path.basename(filename))).groups())]
            for filename in list_of_image_files
        ],
        columns=filename_keys,
    )

    # Fix parsing of src_subject_id.  The filenames do not have an underscore after "NDAR", but src_subject_id values in
    # the data tables do.
    response["src_subject_id"] = [re.sub(r"^NDAR", "NDAR_", subject) for subject in response["src_subject_id"]]
    # Fix parsing of eventname.  The filenames use CamelCase but the datatables use snake_case.
    eventname_conversion: dict[str, str] = {
        "baselineYear1Arm1": "baseline_year_1_arm_1",
        "1YearFollowUpYArm1": "1_year_follow_up_y_arm_1",
        "2YearFollowUpYArm1": "2_year_follow_up_y_arm_1",
        "3YearFollowUpYArm1": "3_year_follow_up_y_arm_1",
        "4YearFollowUpYArm1": "4_year_follow_up_y_arm_1",
    }
    response["eventname"] = [eventname_conversion[event] for event in response["eventname"]]

    return response


def get_data_from_image_files(list_of_files: list[str]) -> list[Any]:
    """
    get_data_from_image_files returns a list of tuples of 4 values.
    The latter 3 values in the tuple are the return from load_nifti.
        <class 'str'>: full file name
        <class 'numpy.ndarray'>: image data as a numpy array
        <class 'numpy.ndarray'>: affine transform as a numpy array
        <class 'nibabel.nifti1.Nifti1Image'>: an object that can be modified and can be written to
               file as a .nii.gz file
    """
    response: list[Any] = [(file, *dipy.io.image.load_nifti(file, return_img=True)) for file in list_of_files]
    return response


def get_white_matter_mask_as_numpy(
    compute_white_matter_mask_from_file: str, mask_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    # The return value is a np.ndarray of bool, a voxel mask that indicates which voxels are to be kept for subsequent
    # analyses.  The mask is flattened to a single dimension because our analysis software indexes voxels this way We
    # determine white matter by looking at the compute_white_matter_mask_from_file for voxels that have an intensity
    # above a threshold.
    source_data = get_data_from_image_files([compute_white_matter_mask_from_file])[0]
    source_voxels: np.ndarray = source_data[1]
    source_affine: np.ndarray = source_data[2]
    white_matter_mask: np.ndarray = (source_voxels >= mask_threshold).reshape(-1)
    print(f"Number of white matter voxels = {np.sum(white_matter_mask)}")
    return white_matter_mask, source_affine


# +
# Functions for reading and selecting data from csv files


def csv_file_to_dataframe(filename: str) -> pd.core.frame.DataFrame:
    """
    A stupid function that reminds us how to read a csv file using pandas
    """
    return pd.read_csv(filename)


def select_rows_of_dataframe(df: pd.core.frame.DataFrame, query_dict: dict[str, str]):
    """
    This function is deprecated in favor of using pandas `isin` functionality.
    """
    # Each key of query_dict is a column header of the df dataframe.  Each value of query_dict is a list of allowed
    # values.  A row will be selected only if each of these columns has one of the allowed keys.
    assert all(key in df.columns for key in query_dict.keys())
    rows: pd.core.frame.DataFrame = df[
        functools.reduce(
            lambda x, y: x & y,
            [
                functools.reduce(lambda x, y: x | y, [df[key] == value for value in values])
                for key, values in query_dict.items()
            ],
        )
    ]
    return rows


# +
# Functions to read and cache KSADS tabular information


def clean_ksads_data_frame(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    KSADS data uses:
        1 = present
        0 = absent
        888 = Question not asked due to primary question response (branching logic)
        555 = Not administered in the assessment
    At least for now, we will assume:
        that 888 means that the question was not asked because the answer was already known to be
            "absent".
        that 555 means we don't know the answer.
    See https://wiki.abcdstudy.org/release-notes/non-imaging/mental-health.html
    """
    for column in df.columns:
        if bool(re.match("ksads_\d", column)):
            df[column] = df[column].astype(float)
            df.loc[df[column] == 555, column] = np.nan
            df.loc[df[column] == 888, column] = 0
    return df


def ksads_filename_to_dataframe(file_mh_y_ksads_ss: str, use_cache: bool = True) -> pd.core.frame.DataFrame:
    """
    Read in the KSADS data, or grab it from cache if the user requests it and we have it.
    """
    rebuild_cache: bool = not use_cache
    try:
        # Check whether we've cached this computation
        df_mh_y_ksads_ss: pd.core.frame.DataFrame = ksads_filename_to_dataframe.df_mh_y_ksads_ss
    except AttributeError:
        # We haven't cached it, so we'll have to read the data even if the user would have permitted us to use the cache
        rebuild_cache = True
    if rebuild_cache:
        print("Begin reading KSADS data file")
        start: float = time.time()
        # Reading from disk takes 10-30 seconds; it is a big file
        df_mh_y_ksads_ss = csv_file_to_dataframe(file_mh_y_ksads_ss)
        # Deal with 555 and 888 values in the data table
        df_mh_y_ksads_ss = clean_ksads_data_frame(df_mh_y_ksads_ss)
        # Place the computed table in the cache
        ksads_filename_to_dataframe.df_mh_y_ksads_ss = df_mh_y_ksads_ss
        print(f"Read KSADS data file in {time.time() - start}s")
        # Return what is now in the cache
    return ksads_filename_to_dataframe.df_mh_y_ksads_ss


# +
# Functions for computing summary statistics for KSADS csv data


def data_frame_value_counts(df: pd.core.frame.DataFrame) -> dict[str, dict[str, np.int64]]:
    """
    Whether an KSADS column of data is interesting depends upon, in part, how much it varies across images.
    Here we perform a census that counts how many times each value occurs in a column.
    """
    # Returns a dict:
    #     Each key is a column name.
    #     Each value is a dict:
    #         Each key of this is a value that occurs in the column.
    #         The corresponding value is the number of occurrences.
    response: dict[str, dict[str, np.int64]] = {
        column: dict(df[column].value_counts(dropna=False).astype(int)) for column in df.columns
    }
    return response


def entropy_of_column_counts(column_counts) -> float:
    """
    Whether an KSADS column of data is interesting depends upon, in part, how much it varies across images.
    Here we compute the entropy (information) of one column given its census data from data_frame_value_counts().
    """
    assert all(value >= 0 for value in column_counts.values())
    total_count = sum(column_counts.values())
    entropy: float = sum(
        [count / total_count * math.log2(total_count / count) for count in column_counts.values() if count > 0]
    )
    return entropy


def ksads_keys_only(all_columns: dict[str, Any]) -> dict[str, Any]:
    """
    This finds those keys that are KSADS variables rather than being index keys (subject, event), etc.
    """
    response: dict[str, Any] = {key: value for key, value in all_columns.items() if bool(re.match("ksads_\d", key))}
    return response


def entropy_of_all_columns(all_columns) -> dict[str, float]:
    """
    Compute entropy (information) of every column by calling subroutine for each column
    """
    response: dict[str, float] = {key: entropy_of_column_counts(value) for key, value in all_columns.items()}
    return response


def find_interesting_entropies(file_mh_y_ksads_ss: str):
    """
    Compute the entropy (information) of each KSADS variable and return them sorted from most entropy to least
    """
    # Find some KSADS data columns with high entropy.
    df_mh_y_ksads_ss: pd.core.frame.DataFrame = ksads_filename_to_dataframe(file_mh_y_ksads_ss)
    counts_for_each_column_mh_y_ksads_ss: dict[str, Any] = ksads_keys_only(data_frame_value_counts(df_mh_y_ksads_ss))
    print("Column counting done")

    entropies: dict[str, float] = entropy_of_all_columns(counts_for_each_column_mh_y_ksads_ss)
    sorted_entropies: dict[str, Any] = dict(sorted(entropies.items(), key=lambda item: item[1], reverse=True))
    # print(f"{sorted_entropies = }", flush=True)
    print("Entropy calculation done")
    return sorted_entropies


def find_interesting_ksads() -> tuple[str, list[str]]:
    """
    The first returned value is the filename for the KSADS data table
    The second returned value is computed via:
        reading in the KSADS data,
        computing the entropy of each column,
        sorting entropies in decreasing order
        choosing just the top few
    or:
        just use values we've computed using this process in the past
    """
    number_ksads_wanted: int = 20  # TODO: Is 20 a good number?
    file_mh_y_ksads_ss: str = "mental-health/mh_y_ksads_ss.csv"
    interesting_ksads: list[str]
    if True:
        full_path = os.path.join(core_directory, file_mh_y_ksads_ss)
        sorted_entropies = find_interesting_entropies(full_path)
        interesting_ksads = list(sorted_entropies.keys())
    else:
        # With "555" and "888" both going to "", the interesting_ksads computation gives:
        interesting_ksads = [
            "ksads_22_142_t",  # Symptom - Insomnia, Past
            "ksads_22_970_t",  # Diagnosis - SLEEP PROBLEMS, Past
            "ksads_2_11_t",  # Symptom - Explosive Irritability, Past
            "ksads_1_2_t",  # Symptom - Depressed Mood, Past
            "ksads_2_13_t",  # Symptom - Decreased Need for Sleep, Past
            "ksads_1_6_t",  # Symptom - Anhedonia, Past
            "ksads_22_141_t",  # Symptom - Insomnia, Present
            "ksads_22_969_t",  # Diagnosis - SLEEP PROBLEMS, Present
            "ksads_2_8_t",  # Symptom - Elevated Mood, Past
            "ksads_10_46_t",  # Symptom - Excessive worries more days than not Past
            "ksads_1_4_t",  # Symptom - Irritability, Past
            "ksads_2_10_t",  # Symptom - ExplosiveIrritability, PresentNext
            "ksads_8_31_t",  # Symptom - Fear of Social Situations, Past
            "ksads_23_146_t",  # Symptom - Wishes/Better off dead, Past
            "ksads_23_957_t",  # Diagnosis - SuicidalideationPassivePast
            "ksads_1_5_t",  # Symptom - Anhedonia, Present
            "ksads_2_839_t",  # Diagnosis - Unspecified Bipolar and Related Disorder, PAST (F31.9)
            "ksads_2_833_t",  # Diagnosis - Bipolar I Disorder, most recent past episode manic (F31.1x)
            "ksads_1_3_t",  # Symptom - Irritability, Present
            "ksads_1_842_t",  # Diagnosis - Major Depressive Disorder, Past (F32.9)
        ]
        # With "555" going to "" and "888" going to "0", the interesting_ksads computation gives:
        interesting_ksads = [
            "ksads_1_187_t",  # Symptom - No two month symptom-free interval, Present
            "ksads_1_188_t",  # Symptom - No two month symptom-free interval, Past
            "ksads_22_142_t",  # Symptom - Insomnia, Past
            "ksads_22_970_t",  # Diagnosis - SLEEP PROBLEMS, Past
            "ksads_2_11_t",  # Symptom - Explosive Irritability, Past
            "ksads_2_222_t",  # Symptom - Lasting at least 4 days, Past
            "ksads_1_184_t",  # Symptom - Impairment in functioning due to depression, Past
            "ksads_1_2_t",  # Symptom - Depressed Mood, Past
            "ksads_2_13_t",  # Symptom - Decreased Need for Sleep, Past
            "ksads_1_6_t",  # Symptom - Anhedonia, Past
            "ksads_1_160_t",  # Symptom - Fatigue, Past
            "ksads_1_162_t",  # Symptom - Concentration Disturbance, Past
            "ksads_2_220_t",  # Symptom - Lasting at least one week, Past
            "ksads_1_156_t",  # Symptom - Insomnia when depressed, Past
            "ksads_1_174_t",  # Symptom - Psychomotor Agitation in Depressive Disorder, Past
            "ksads_2_216_t",  # Symptom - Impairment in functioning due to bipolar, Past
            "ksads_2_208_t",  # Symptom - Psychomotor Agitation in Bipolar Disorder, Past
            "ksads_2_206_t",  # Symptom - Increased Energy, Past
            "ksads_22_141_t",  # Symptom - Insomnia, Present
            "ksads_22_969_t",  # Diagnosis - SLEEP PROBLEMS, Present
        ]

    interesting_ksads = interesting_ksads[:number_ksads_wanted]
    ksads_vars: tuple[str, list[str]] = (file_mh_y_ksads_ss, interesting_ksads)
    return ksads_vars


# +
# Find all images for which we have tabular data


# confounding_vars_config is a dictionary organized by Fieldname.
# In the output, fieldnames will be extened in cases where multicolumn representations are required, such as for one-hot
#     columns for unordered data.
# The value associated with a fieldname key is itself a dictionary.  These keys are required:
#     "File": Required.  Location within core_directory to read data from.
#     "HandleMissing": Required.
#         "invalidate" (throw away scan if it has this field missing)
#         "together" (all scans marked as "missing" are put in one category that is dedicated to all missing data)
#         "byValue" (for each IsMissing value, all scans with that value are put in a category)
#         "separately" (each row with a missing value is its own category; e.g., a patient with no siblings in the
#             study)
#     "Type":  Required.
#         "ordered" (can be interpretted as a meaninful number) or
#         "unordered" (each distinct but one value gets a one-hot column)
# These keys are optional:
#     "InternalName": name used within "File" for this fieldname.  Defaults to the fieldname itself.
#     "Convert": a Python dictionary that shows how to translate keys (e.g., change from NaN, "777") to values (e.g.,
#         changed to "").
#     "IsMissing": a list of values that, after conversion, should be considered missing.  Defaults to ["", NaN].
#     "Longitudinal": list that contains one or more of the following.  Default is ["intercept"].
#         "time": field is a date, or possibly a sequence number.  At most one fieldname can be "time".
#         "intercept": field is used as a normal number (ordered) or a normal one-hot (unordered).
#         "slope": also add column(s) like "intercept" but with values multiplied by "time".  A fieldname with both
#             "time" and "slope" will produce quadradic time values in a dataframe column.


def large_enough_category_size(df_var: pd.core.frame.DataFrame, min_category_size: int) -> list[bool]:
    response: list[bool] = [
        df_var[column].nunique() != 2 or df_var[column].value_counts().min() >= min_category_size
        for column in df_var.columns
    ]
    return response


def process_confounding_var(
    fieldname: str, details: dict[str, Any], join_keys: list[str], min_category_size: int
) -> pd.core.frame.DataFrame:
    mesgs: list[str] = []
    if "File" not in details:
        mesgs.append(f"The 'File' attribute must be supplied for the fieldname {fieldname!r}.")
    if "HandleMissing" not in details:
        mesgs.append(f"The 'HandleMissing' attribute must be supplied for the fieldname {fieldname!r}.")
    if "Type" not in details:
        mesgs.append(f"The 'Type' attribute must be supplied for the fieldname {fieldname!r}.")
    if mesgs:
        mesg = "\n".join(mesgs)
        raise ValueError(mesg)

    Fieldname: str = fieldname
    File: str = details["File"]
    HandleMissing: str = details["HandleMissing"]
    Type: str = details["Type"]
    InternalName: str = details.get("InternalName", Fieldname)
    Convert: dict[Any, Any] = details.get("Convert", ConvertDefault)
    IsMissing: dict[Any, Any] = details.get("IsMissing", IsMissingDefault)
    Longitudinal: dict[Any, Any] = details.get("Longitudinal", LongitudinalDefault)

    mesgs = []
    if not all(
        key in {"File", "HandleMissing", "Type", "InternalName", "Convert", "IsMissing", "Longitudinal"}
        for key in details.keys()
    ):
        mesgs.append(f"At least one of {list(details.keys())!r} is not a valid name for {fieldname!r}.")
    if HandleMissing not in {"invalidate", "together", "byValue", "separately"}:
        mesgs.append(f"`{details['HandleMissing']!r}` is not a valid HandleMissing value for {fieldname!r}.")
    if Type not in {"ordered", "unordered"}:
        mesgs.append(f"`{details['Type']!r}` is not a valid Type value for {fieldname!r}.")
    if len(Longitudinal) == 0 or not all(element in {"time", "intercept", "slope"} for element in Longitudinal):
        mesgs.append(f"`{details['Longitudinal']!r}` is not a valid Longitudinal value for {fieldname!r}.")
    if mesgs:
        mesg = "\n".join(mesgs)
        raise ValueError(mesg)

    # Read the desired data column and join keys from file
    df_var: pd.core.frame.DataFrame = csv_file_to_dataframe(os.path.join(core_directory, File))[
        [*join_keys, InternalName]
    ]
    # print(f"  #01 {df_var.columns = }", flush=True)

    # Rename the InternalName column to Fieldname
    if InternalName != Fieldname:
        df_var = df_var.rename(columns={InternalName: Fieldname})
        # print(f"  #02 {df_var.columns = }", flush=True)

    # Perform conversions of values
    for src, dst in Convert.items():
        df_var[Fieldname] = df_var[Fieldname].replace(src, dst)
        # print(f"    #03 {src} -> {dst}: {df_var.columns = }", flush=True)

    # Do the right thing with missing values
    if HandleMissing != "byValue" and len(IsMissing) > 1:
        # Make all missing values equal to IsMissing[0]
        for val in IsMissing[1:]:
            df_var[Fieldname] = df_var[Fieldname].replace(val, IsMissing[0])
            # print(f"    #04 {val = }: {df_var.columns = }", flush=True)
        IsMissing = IsMissing[:1]
    if HandleMissing == "invalidate" and len(IsMissing):
        # Remove rows for missing values
        df_var = df_var[df_var[Fieldname] != IsMissing[0]]
        # print(f"  #05 {df_var.columns = }", flush=True)
    if HandleMissing == "together":
        if Type == "unordered":
            # There is nothing more to do.
            pass
        if Type == "ordered":
            # We will interpret missing as 0, but add a one-hot column so that it can effectively be any constant.  In
            # particular, it will not be confounded with actual values of 0.
            new_missing_name = Fieldname + "_missing"
            if new_missing_name in df_var.columns:
                mesg = f"Failed to get unique column name for {new_missing_name!r}."
                raise ValueError(mesg)
            df_var[new_missing_name] = (df_var[Fieldname] == IsMissing[0]).astype(int)
            # print(f"  #06 {df_var.columns = }", flush=True)
            df_var.loc[df_var[Fieldname] == IsMissing[0], Fieldname] = 0
            # print(f"  #07 {df_var.columns = }", flush=True)
    if HandleMissing == "byValue":
        if Type == "unordered":
            # There is nothing more to do.
            pass
        if Type == "ordered":
            # TODO: Do we need to implement this case?
            mesg = 'We do not currently handle the case that HandleMissing == "byValue" and Type == "ordered"'
            raise ValueError(mesg)
    if HandleMissing == "separately":
        # We want to use unused distinct values for each type of "missing".  We add unique values so the pd.get_dummies
        # call creates a one-hot column for each missing value.
        unused_numeric_value = 1 + max([int(x) for x in df_var[Fieldname] if isinstance(x, (int, float))] + [0])
        number_needed_values = (df_var[Fieldname] == IsMissing[0]).sum()
        if Type == "unordered":
            df_var.loc[df_var[Fieldname] == IsMissing[0], Fieldname] = range(
                unused_numeric_value, unused_numeric_value + number_needed_values
            )
            # print(f"  #08 {df_var.columns = }", flush=True)

        if Type == "ordered":
            # TODO: Do we need to implement this case?
            mesg = 'We do not currently handle the case that HandleMissing == "separately" and Type == "ordered"'
            raise ValueError(mesg)

    # Convert categorical data to a multicolumn one-hot representation
    if Type == "unordered":
        df_var = pd.get_dummies(df_var, dummy_na=True, columns=[Fieldname], drop_first=False)
        # print(f"  #09 {df_var.columns = }", flush=True)

    # Remove columns that are constant
    df_var = df_var.loc[:, (df_var.nunique() > 1) | df_var.columns.isin(join_keys)]
    # print(f"  #10 {df_var.columns = }", flush=True)
    # Remove categories with too few members
    df_var = df_var.loc[:, large_enough_category_size(df_var, min_category_size) | df_var.columns.isin(join_keys)]
    # print(f"  #11 {df_var.columns = }", flush=True)

    print(f"process_confounding_var({fieldname!r}) gives {df_var.columns = }")
    return df_var


def process_longitudinal_config(
    confounding_vars_config: dict[str, dict[str, Any]],
    df_dict: dict[str, pd.core.frame.DataFrame],
    join_keys: list[str],
) -> list[pd.core.frame.DataFrame]:
    # TODO: Do we need to worry about "time" and "slope" with target_vars and tested_vars too?

    # The value associated with each "Longitudinal" key is a list of zero or more of "time", "intercept", and "slope";
    # the default is LongitudinalDefault.  If the list is empty, the fieldname should not be output!  A fieldname that
    # includes "intercept" in its "Longitudinal" list will be passed through unmodified.  The unique fieldname that
    # includes "time" in its "Longitudinal" list will be used in the creation of data columns for those fieldnames that
    # include "slope" in their "Longitudinal" lists.
    has_time = [
        fieldname
        for fieldname, details in confounding_vars_config.items()
        for longitudinal in [details.get("Longitudinal", LongitudinalDefault)]
        if "time" in longitudinal
    ]
    has_intercept = [
        fieldname
        for fieldname, details in confounding_vars_config.items()
        for longitudinal in [details.get("Longitudinal", LongitudinalDefault)]
        if "intercept" in longitudinal
    ]
    has_slope = [
        fieldname
        for fieldname, details in confounding_vars_config.items()
        for longitudinal in [details.get("Longitudinal", LongitudinalDefault)]
        if "slope" in longitudinal
    ]
    # print(f"  {has_time = }", flush=True)
    # print(f"  {has_intercept = }", flush=True)
    # print(f"  {has_slope = }", flush=True)

    mesgs: list[str] = []
    if len(has_time) > 1:
        mesgs.append(
            f'{len(has_time)} confounding_var fields {has_time} were specified as "time" but more than 1 is not '
            "permitted."
        )
    if len(has_time) == 1 and len(has_slope) == 0:
        mesgs.append(
            f'When one confounding_var fieldname {has_time} is specified as "time" then at least one must be '
            'specified as "slope".'
        )
    if len(has_time) == 0 and len(has_slope) > 0:
        mesgs.append(
            f'{len(has_slope)} confounding_var fields were supplied as "slope" {has_slope} but none are permitted '
            'because no confounding fields are supplied as "time".'
        )
    if mesgs:
        mesg = "\n".join(mesgs)
        raise ValueError(mesg)

    # We return one dataframe for each fieldname in has_intercept.
    df_intercepts: list[pd.core.frame.DataFrame] = [df_dict[key] for key in has_intercept]
    # print(f"  {set([col for df in df_intercepts for col in df.columns]) = }", flush=True)

    # If we have a "time" variable then we add a dataframe for each fieldname in has_slope (even if fieldname is also in
    # has_intercept).  Note that if df_time also has "slope" then we will get quadratic values in a column.
    df_slopes: list[pd.core.frame.DataFrame] = []
    if len(has_time) == 1 and len(has_slope) > 0:
        df_time: pd.core.frame.DataFrame = df_dict[has_time[0]].copy(deep=True)
        new_time_name: str = "ubhzaeZTE3McmbxX"
        if new_time_name in df_time.columns:
            mesg = "Failed to get unique column name for df_time"
            raise ValueError(mesg)
        df_time = df_time.rename(columns={has_time[0]: new_time_name})
        for fieldname in has_slope:
            df_slope: pd.core.frame.DataFrame = df_dict[fieldname].copy(deep=True)
            if new_time_name in df_slope.columns:
                mesg = "Failed to get unique column name for df_slope"
                raise ValueError(mesg)

            df_merged: pd.core.frame.DataFrame
            df_merged = pd.merge(df_time, df_slope, on=join_keys, how="inner", validate="one_to_one")

            columns_to_process: list[str] = list(set(df_slope.columns).difference(set(join_keys)))
            for column in columns_to_process:
                new_slope_name: str = column + "_slope"
                if new_slope_name in df_merged.columns:
                    mesg = f"Failed to get unique column name for {new_slope_name!r}."
                    raise ValueError(mesg)
                df_merged[new_slope_name] = df_merged[column] * df_merged[new_time_name]
                df_merged = df_merged.drop(columns=[*columns_to_process, new_time_name], axis=1)

            df_slopes.append(df_merged)
            # print(f"  {set([col for df in df_slopes for col in df.columns]) = }", flush=True)

    return df_intercepts + df_slopes


def make_dataframe_for_confounding_vars(
    confounding_vars_config: dict[str, dict[str, Any]], join_keys: list[str], min_category_size: int
) -> pd.core.frame.DataFrame:
    df_dict: dict[str, pd.core.frame.DataFrame] = {
        fieldname: process_confounding_var(fieldname, details, join_keys, min_category_size)
        for fieldname, details in confounding_vars_config.items()
    }
    # print(f"  {df_dict.keys() = }", flush=True)
    df_list: list[pd.core.frame.DataFrame] = process_longitudinal_config(confounding_vars_config, df_dict, join_keys)
    # print(f"  {set([col for df in df_list for col in df.columns]) = }", flush=True)
    df_all_keys: pd.core.frame.DataFrame = df_list[0]
    # print(f"  #01 {set(df_all_keys.columns) = }", flush=True)
    for df_next in df_list[1:]:
        # print(f"    #02 {set(df_all_keys.columns) = }", flush=True)
        # print(f"    #03 {set(df_next.columns) = }", flush=True)
        df_all_keys = pd.merge(df_all_keys, df_next, on=join_keys, how="inner", validate="one_to_one")
        # print(f"    #04 {set(df_all_keys.columns) = }", flush=True)
        # print(f"  {df_all_keys.columns = }", flush=True)
    return df_all_keys


def merge_confounding_table(
    confounding_vars_config: dict[str, dict[str, Any]],
    coregistered_images_directory: str,
    join_keys: list[str],
    min_category_size: int,
) -> tuple[pd.core.frame.DataFrame, list[str]]:
    """
    This creates the master data table from disparate sources that includes confounding variables and image meta data.
    It also returns a list of the confounding variable names

    Create a dataframe that contains all confounding variables (regardless of their source data table)
    Create a dataframe that contains meta data about the available images
    Merge these two dataframes using the join_keys
    Note that a (src_subject_id, eventname) pair can occur more than once, e.g., for both "md" and "fa" image subtypes
    """
    df_all_keys: pd.core.frame.DataFrame
    df_all_keys = make_dataframe_for_confounding_vars(confounding_vars_config, join_keys, min_category_size)
    confounding_keys: list[str] = sorted(set(df_all_keys.columns).difference(set(join_keys)))

    list_of_image_files: list[str] = get_list_of_image_files(coregistered_images_directory)
    df_image_information: pd.core.frame.DataFrame = parse_image_filenames(list_of_image_files)
    df_all_images: pd.core.frame.DataFrame = pd.merge(
        df_all_keys,
        df_image_information[[*join_keys, "image_subtype", "filename"]],
        on=join_keys,
        how="inner",
        validate="one_to_many",
    )
    return df_all_images, confounding_keys


# +
def find_good_slice(margins):
    """
    For each ksads variable, we want to show an interesting slice of the 3d-data.
    For example, we choose a slice in the X dimension (i.e., we choose a YZ plane) by designating its x coordinate and
        its lower and upper bounds for y and z.
    First step is summing out over the Y and Z dimensions to compute `margins`, which is done before calling
        find_good_slice().
    (`margins` is 2-dimensional; it is computed from 4-dimensional data with shape=(number_tested_variables, size_x,
        size_y, size_z))
    We then compute:
        for i in range(list_of_tested_variables):
            min_[i] = The lowest x for which margins[i, x] is non-zero
            max_[i] = One more than the largest x for which margins[i, x] is non-zero
            best_[i] = The (first) value of x that maximizes margins[i, :]
    This routine works identically for slices in the Y or Z dimensions, so long as margins is supplied by summing out
    the remaining dimensions.
    """
    min_ = np.argmax(margins > 0.0, axis=-1)
    best_ = np.argmax(margins, axis=-1)
    max_ = np.argmax(np.cumsum(margins > 0.0, axis=-1), axis=-1) + 1
    return min_, best_, max_


def find_local_maxima(voxels: np.ndarray, threshold: float, order: int) -> list[tuple[int, int, int]]:
    """
    Find the locations of each voxels whose intensity value exceeds or equals all its neighbors.  Another voxel is a
    neighbor if each of its three coordinates is within `order` in value.
    """
    shapex, shapey, shapez = voxels.shape
    maxima: list[tuple[int, int, int]] = [
        (x, y, z)
        for x, y, z in {
            (x, y, int(z))
            for x in range(shapex)
            for y in range(shapey)
            for z in scipy.signal.argrelextrema(voxels[x, y, :], np.greater_equal, order=order)[0]
        }
        | {
            (x, int(y), z)
            for x in range(shapex)
            for z in range(shapez)
            for y in scipy.signal.argrelextrema(voxels[x, :, z], np.greater_equal, order=order)[0]
        }
        | {
            (int(x), y, z)
            for y in range(shapey)
            for z in range(shapez)
            for x in scipy.signal.argrelextrema(voxels[:, y, z], np.greater_equal, order=order)[0]
        }
        if voxels[x, y, z] > threshold
        for neighborhood in [
            voxels[
                max(0, x - order) : min(shapex, x + order + 1),
                max(0, y - order) : min(shapey, y + order + 1),
                max(0, z - order) : min(shapez, z + order + 1),
            ]
        ]
        if voxels[x, y, z] == np.max(neighborhood)
    ]
    # Sort from highest intensity to lowest
    maxima.sort(key=lambda xyz: -voxels[xyz[0], xyz[1], xyz[2]])
    return maxima


# +
def show_maximum_using_partition(
    xyz: tuple[int, int, int],
    log10_pvalue: np.ndarray,
    fa_partition_voxels: np.ndarray,
    fa_partition_lookup: dict[int, str],
):
    x, y, z = xyz
    shapex, shapey, shapez = log10_pvalue.shape
    region_index = fa_partition_voxels[x, y, z]
    where = "in"
    if region_index == background_index:
        # Instead of background, take the most common nearby brain region
        for distance in range(1, 4):
            neighborhood = fa_partition_voxels[
                max(0, x - distance) : min(shapex, x + distance + 1),
                max(0, y - distance) : min(shapey, y + distance + 1),
                max(0, z - distance) : min(shapez, z + distance + 1),
            ].reshape(-1)
            neighborhood = neighborhood[neighborhood != background_index].astype(int)
            if neighborhood.size:
                values, counts = np.unique(neighborhood, return_counts=True)
                # Ties go to the lower integer; oh well
                region_index = values[np.argmax(counts)]
                where = "near"
                break
    print(
        f"Found local maximum -log_10 p(t-stat)={round(1000.0 * log10_pvalue[x, y, z]) / 1000.0}"
        f" at ({x}, {y}, {z} = {shapez - 1}-{shapez - 1 - z})"
        f" {where} region {fa_partition_lookup[region_index]} ({region_index})"
    )


def show_maximum_using_cloud(
    xyz: tuple[int, int, int],
    log10_pvalue: np.ndarray,
    fa_cloud_voxels: np.ndarray,
    fa_cloud_lookup: dict[int, str],
    threshold: float = 0.25,
):
    x, y, z = xyz
    shapex, shapey, shapez = log10_pvalue.shape
    cloud_here = fa_cloud_voxels[:, x, y, z]
    argsort: np.ndarray = np.argsort(cloud_here)[::-1]
    print(
        f"Found local maximum -log_10 p(t-stat)={round(1000.0 * log10_pvalue[x, y, z]) / 1000.0}"
        f" at ({x}, {y}, {z} = {shapez - 1}-{shapez - 1 - z} in regions:"
    )
    # Show those that exceed threshold; showing at least 2 regions
    for cloud_index in [argsort[i] for i in range(len(argsort)) if i < 2 or cloud_here[argsort[i]] >= threshold]:
        region = cloud_index + background_index + 1
        print(
            f"    {fa_cloud_lookup[region]} ({region}) confidence = {round(100000.0 * cloud_here[cloud_index]) / 1000}%"
        )


# How we might process the inputs using nilearn.mass_univariate.permuted_ols()
def use_nilearn(
    confounding_keys: list[str],
    confounding_table: pd.core.frame.DataFrame,
    tested_keys: list[str],
    tested_table: pd.core.frame.DataFrame,
    join_keys: list[str],
    min_category_size: int,
    white_matter_mask: nibabel.nifti1.Nifti1Image,
) -> dict[str, dict[str, np.ndarray]]:
    print(f"Using nilearn version {nilearn.__version__}")
    # print(f"{white_matter_mask = }")
    # # confounding_table has columns *counfounding_vars, src_subject_id, eventname, image_subtype, filename
    # print(f"{confounding_table = }")
    # print(f"{tested_keys = }")  # list of str of selected ksads
    # print(f"{tested_table = }")  # Dataframe across all ksads
    # print(f"{confounding_keys = }")  # list of str of confounding keys

    # Find "fa" and "md" and any others that are added later
    image_subtypes = list(confounding_table["image_subtype"].unique())

    all_subtypes: dict[str, dict[str, np.ndarray]] = {}
    # For each subtype (e.g., "fa" and "md")
    for image_subtype in image_subtypes:
        print(f"{image_subtype = }")

        # Create a single data table that includes all confounding_variables and all ksads variables
        all_information: pd.core.frame.DataFrame = pd.merge(
            confounding_table[confounding_table["image_subtype"] == image_subtype],
            tested_table[[*join_keys, *tested_keys]],
            on=join_keys,
            how="inner",
            validate="one_to_one",
        )
        # Don't include any images for which we are missing variable information
        all_information = all_information.dropna()

        """
        Although pandas is a great way to read and manipulate these data tables, nilearn expects them to be
        "array-like".  Because panda tables require `.iloc` to accept integer coordinates for a 2d-table, panda tables
        fail "array-like".  We'll use numpy arrays.

        See https://nilearn.github.io/stable/modules/generated/nilearn.mass_univariate.permuted_ols.html
        Create the three main inputs to nilearn.mass_univariate.permuted_ols() as numpy arrays
        """

        tested_input_pandas: pd.core.frame.DataFrame = all_information[tested_keys]
        print(f"  {tested_input_pandas.columns = }")
        tested_input: np.ndarray = tested_input_pandas.to_numpy(dtype=float)
        print(f"  {tested_input.shape = }")

        confounding_input_pandas: pd.core.frame.DataFrame = all_information[confounding_keys]
        # print(f"  {confounding_input_pandas.columns = } (penultimate)")
        confounding_input_pandas = confounding_input_pandas.loc[
            :,
            large_enough_category_size(confounding_input_pandas, min_category_size)
            | confounding_input_pandas.columns.isin(join_keys),
        ]
        print(f"  {confounding_input_pandas.columns = }", flush=True)
        confounding_input: np.ndarray = confounding_input_pandas.to_numpy(dtype=float)
        print(f"  {confounding_input.shape = }")

        white_matter_indices: np.ndarray = (white_matter_mask_input.get_fdata() != 0.0).reshape(-1)
        target_input: np.ndarray = np.stack(
            [
                np_voxels.reshape(-1)[white_matter_indices]
                for a, np_voxels, c, d in get_data_from_image_files(list(all_information["filename"].values))
            ]
        )
        print(f"  {target_input.shape = }")

        # Set all other input parameters for nilearn.mass_univariate.permuted_ols()
        model_intercept: bool = True
        n_perm: int = 10000
        two_sided_test: bool = True
        random_state = None
        n_jobs: int = -1  # All available
        verbose: int = 1
        # The white_matter_mask (a nibabel.nifti1.Nifti1Image) is not used directly.  Wrap it and help it appropriately:
        masker = nilearn.maskers.NiftiMasker(white_matter_mask)
        masker.fit()
        # Some web pages say that we must invoke masker.transform in order to have masker.inverse_transform available
        # within nilearn.
        _ = masker.transform(white_matter_mask)
        # TODO: Should tfce be True here, or instead at a second-level analysis (using non_parametric_inference).
        tfce: bool = False
        # TODO: How should we set `threshold`.  (For TFCE==True, it should be `None`.)
        threshold = None
        output_type: str = "dict"

        print(f"  {n_perm = }")
        print(f"  {tfce = }")
        sys.stdout.flush()

        # Ask nilearn to compute our p-values, and apply tfce if tfce==True.
        # TODO: Consider using nilearn.glm.second_level.non_parametric_inference() instead
        response: dict[str, np.ndarray] = nilearn.mass_univariate.permuted_ols(
            tested_vars=tested_input,  # ksads
            target_vars=target_input,  # voxels
            confounding_vars=confounding_input,  # e.g., interview_age
            model_intercept=model_intercept,
            n_perm=n_perm,
            two_sided_test=two_sided_test,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            masker=masker,
            tfce=tfce,
            threshold=threshold,
            output_type=output_type,
        )
        # Record the response of nilearn for this subtype (e.g., "fa" or "md")
        all_subtypes[image_subtype] = response
    return all_subtypes


# -

# ## Define or load input data

# +
# Set inputs for our task.  Ultimately these will either be class members or parameters for class methods.

confounding_vars_config: dict[str, dict[str, Any]] = {
    "interview_age": {
        "File": "abcd-general/abcd_y_lt.csv",
        "HandleMissing": "invalidate",
        "Type": "ordered",
        "Longitudinal": ["intercept"],
    },
    # "site_id_l": {
    #     "File": "abcd-general/abcd_y_lt.csv",
    #     "HandleMissing": "invalidate",
    #     "Type": "unordered",
    #     "Longitudinal": ["intercept"],
    # },
    "demo_gender_id_v2": {
        "File": "gender-identity-sexual-health/gish_p_gi.csv",
        "Convert": {"777": "", "999": ""},
        "HandleMissing": "invalidate",
        "Type": "unordered",
        "Longitudinal": ["intercept"],
    },
    # "rel_family_id": {
    #     "File": "abcd-general/abcd_y_lt.csv",
    #     "HandleMissing": "together",
    #     "Type": "unordered",
    # },
}

# confounding_vars_input: list[tuple[str, list[str]]] = [
#     (
#         "physical-health/ph_y_bld.csv",
#         [
#             # TODO: Can any of these be useful, or are they sparse/empty of useful information?
#             # "biospec_blood_baso_percent",  # BASO %
#             # "biospec_blood_baso_abs",  # BASO ABS
#             # "biospec_blood_eos_percent",  # EOS %
#             # "biospec_blood_eos_abs",  # EOS ABS
#             # "biospec_blood_hemoglobin",  # Hemoglobin
#             # "biospec_blood_mcv",  # MCV
#             # "biospec_blood_plt_count",  # PLT Count
#             # "biospec_blood_wbc_count",  # WBC Count
#             # "biospec_blood_ferritin",  # Ferritin
#             # "biospec_blood_hemoglobin_a1",  # hemoglobin_a1
#             # "biospec_blood_imm_gran_per",  # Immature Gran %
#         ],
#     ),
# ]
# +
# Load tabular information for counfounding variables
start = time.time()
confounding_table_input, confounding_keys_input = merge_confounding_table(
    confounding_vars_config, coregistered_images_directory, join_keys_global, min_category_size_global
)
print(f"{confounding_keys_input = }")
print(f"{len(confounding_table_input) = }")
print(f"Total time to load confounding information = {time.time() - start}s")

# Load KSADS variables information.  This is slow if not cached.
file_mh_y_ksads_ss_input, tested_keys_input = find_interesting_ksads()
tested_table_input = ksads_filename_to_dataframe(file_mh_y_ksads_ss_input)
# -
# ## Run the workflow


# +
# See https://nilearn.github.io/dev/modules/generated/nilearn.masking.compute_brain_mask.html
target_img = nibabel.load(compute_white_matter_mask_from_file)
white_matter_mask_affine = target_img.affine
white_matter_mask_input = nilearn.masking.compute_brain_mask(
    target_img=target_img,
    threshold=mask_threshold_global,
    connected=False,  # TODO: Is this best?
    opening=False,  # False means no image morphological operations.  An int represents an amount of it.
    memory=None,
    verbose=2,
    mask_type="wm",  # "whole-brain", "gm" (gray matter), "wm" (white matter)
)
print(f"Number of white_matter voxels = {np.sum(white_matter_mask_input.get_fdata())}")

if True:
    # Move demo_gender_id_v2_1.0 from confounding_table_input to tested_table_input.  Note that confounding_table_input
    # has subtype information but tested_table_input does not.
    var_prefix = "demo_gender_id_v2_"
    var_name = var_prefix + "1.0"
    if var_name in confounding_table_input.columns:
        print(f"{confounding_table_input.columns = }")
        tested_keys_input = [var_name, *tested_keys_input]
        tested_table_input = pd.merge(
            tested_table_input,
            confounding_table_input[[*join_keys_global, var_name]].drop_duplicates(),
            on=join_keys_global,
            how="inner",
            validate="one_to_one",
        )
        confounding_keys_input = [key for key in confounding_keys_input if not key.startswith(var_prefix)]
        confounding_table_input = confounding_table_input.drop(
            columns=[col for col in confounding_table_input.columns if col.startswith(var_prefix)]
        )
else:
    # TODO: Do we need to set var_name to something meaningful in this case too?
    var_name = ""


start = time.time()
output_voxels_by_subtype: dict[str, dict[str, np.ndarray]] = use_nilearn(
    confounding_keys_input,
    confounding_table_input,
    tested_keys_input,
    tested_table_input,
    join_keys_global,
    min_category_size_global,
    white_matter_mask_input,
)
print(f"Computed all voxels in time {time.time() - start}s")
# -


# ## Show some output

# Show both image types: ['fa', 'md']
print(f"{list(output_voxels_by_subtype.keys()) = }")
# Show returned data types: ['t', 'logp_max_t', 'h0_max_t']
print(f"{list(output_voxels_by_subtype['fa'].keys()) = }")

# We used use_nilearn().  Show some values that might help us to sanity check these outputs.  nilearn returned only
# voxels in the white matter, so we construct images that include background.
output_images_by_subtype: dict[str, np.ndarray] = {}
white_matter_indices: np.ndarray = (white_matter_mask_input.get_fdata() > 0).reshape(-1)
print("## use_nilearn output")
# Set gamma to, e.g., 0.1 or 0.01 to change the contrast of the image.  Lower values of gamma brighten the darkest
# voxels the most.
gamma = 0.01
if gamma != 1.0:
    print(f"Using contrast adjustment with {gamma = }")
for subtype_key, subtype_value in output_voxels_by_subtype.items():
    print(f"## {subtype_key = }")
    for table in subtype_value.keys():
        # The three types of table are ['t', 'logp_max_t', 'h0_max_t'].  We're probably most interested in
        # 'logp_max_t'.
        print(f"#### table: {table = }")
        # Show the shape of this output
        print(f"output_voxels_by_subtype[{subtype_key!r}][{table!r}].shape = {subtype_value[table].shape}")
        # Show the sum of all values of this output
        print(f"np.sum(output_voxels_by_subtype[{subtype_key!r}][{table!r}]) = {np.sum(subtype_value[table])}")
        # Count how many of the values are not NaNs.
        print(
            f"np.sum(~np.isnan(output_voxels_by_subtype[{subtype_key!r}][{table!r}])) = "
            f"{np.sum(~np.isnan(subtype_value[table]))}"
        )
        # Ask to see the whole table; though Python cuts out much of it
        # print(
        #     f"output_voxels_by_subtype[{subtype_key!r}][{table!r}] = "
        #     f"{subtype_value[table]}"
        # )
    print("#### image information")
    number_tested_vars = subtype_value["logp_max_t"].shape[0]
    # We will make an output image for each tested variable (i.e., each KSADS variable).  Background is zeros.
    output_images_for_subtype: np.ndarray
    output_images_for_subtype = np.zeros((number_tested_vars, *white_matter_mask_input.get_fdata().shape))
    # Forground is from nilearn
    output_images_for_subtype.reshape(number_tested_vars, -1)[:, white_matter_indices] = subtype_value["logp_max_t"]
    # Stash the image into a dictionary that we are building
    output_images_by_subtype[subtype_key] = output_images_for_subtype
    # Show the shape of the image
    print(f"output_images_for_subtype[{subtype_key!r}].shape = {output_images_for_subtype.shape}")
    # Show the minimum voxel intensity
    print(f"np.min(output_images_for_subtype[{subtype_key!r}]) = {np.min(output_images_for_subtype)}")
    # Show the maximum voxel intensity
    print(f"np.max(output_images_for_subtype[{subtype_key!r}]) = {np.max(output_images_for_subtype)}")

    # Find interestig slices to plot.  For each tested variable (i.e., each KSADS variable), we'll have one X slice,
    # one Y slice, and one Z slice.
    x_margins = output_images_for_subtype.sum(axis=(2, 3))
    minX, bestX, maxX = find_good_slice(x_margins)

    y_margins = output_images_for_subtype.sum(axis=(1, 3))
    minY, bestY, maxY = find_good_slice(y_margins)

    z_margins = output_images_for_subtype.sum(axis=(1, 2))
    minZ, bestZ, maxZ = find_good_slice(z_margins)

    faYes: bool = subtype_key == "fa"
    if faYes:
        # Load in reference segmentation (partition)
        fa_partition_data: dict[str, Any] = slicerio.read_segmentation(fa_partition_filename)
        fa_partition_voxels: np.ndarray = fa_partition_data["voxels"].astype(np.int32)
        fa_partition_lookup: dict[int, str] = {d["labelValue"]: d["name"] for d in fa_partition_data["segments"]}
        del fa_partition_data
        # Load in reference segmentation (cloud)
        fa_cloud_data: dict[str, Any] = slicerio.read_segmentation(fa_cloud_filename)
        fa_cloud_voxels: np.ndarray = fa_cloud_data["voxels"].astype(np.float64)
        fa_cloud_lookup: dict[int, str] = {d["labelValue"]: d["name"] for d in fa_cloud_data["segments"]}
        del fa_cloud_data
        # Load in brain for output background
        brain_data: dict[str, Any] = slicerio.read_segmentation(compute_white_matter_mask_from_file)
        brain_voxels: np.ndarray = brain_data["voxels"].astype(np.float64)
        brain_voxels = brain_voxels / np.max(brain_voxels)  # max is now 1, to scale to -log10(10%)==1
        # Change brain_voxels to gray in RGB.  Matplotlib expects color to be the last dimension
        brain_voxels = np.stack((brain_voxels,) * 3, axis=-1)
        del brain_data

    print()
    for i in range(number_tested_vars):
        output_image: np.ndarray = output_images_for_subtype[i, :, :, :]
        if faYes and tested_keys_input[i] == var_name:
            dipy.io.image.save_nifti(an_output_filename, output_image, white_matter_mask_affine)

        if faYes:
            maxima: list[tuple[int, int, int]] = find_local_maxima(output_image, 0.1, 3)
            print("#### Maxima information")
            print(f"{subtype_key!r} image for {tested_keys_input[i]!r}")
            for xyz in maxima:
                show_maximum_using_partition(xyz, output_image, fa_partition_voxels, fa_partition_lookup)
            for xyz in maxima:
                show_maximum_using_cloud(xyz, output_image, fa_cloud_voxels, fa_cloud_lookup, threshold=0.1)
        import IPython

        shell = IPython.get_ipython()
        if shell and shell.__class__.__name__ == "ZMQInteractiveShell":
            shapex, shapey, shapez = output_image.shape
            # Run this code if within Jupyter
            # For each tested variable (i.e., each KSADS variable), we'll have one X slice, one Y slice, and one Z
            # slice.
            print(f"{subtype_key!r} image X={bestX[i]} slice for {tested_keys_input[i]!r}")
            slice_2d = brain_voxels[bestX[i], :, :, :].copy() if faYes else np.zeros((shapey, shapez, 3))
            # Add gamma corrected output to the green channel
            slice_2d[:, :, 1] += np.power(output_image[bestX[i], :, :], gamma)
            slice_2d = np.clip(slice_2d, 0, 1)
            matplotlib.pyplot.imshow(slice_2d)
            # matplotlib.pyplot.colorbar()
            matplotlib.pyplot.show()

            print(f"{subtype_key!r} image Y={bestY[i]} slice for {tested_keys_input[i]!r}")
            slice_2d = brain_voxels[:, bestY[i], :, :].copy() if faYes else np.zeros((shapex, shapez, 3))
            # Add gamma corrected output to the green channel
            slice_2d[:, :, 1] += np.power(output_image[:, bestY[i], :], gamma)
            slice_2d = np.clip(slice_2d, 0, 1)
            matplotlib.pyplot.imshow(slice_2d, cmap="gray")
            # matplotlib.pyplot.colorbar()
            matplotlib.pyplot.show()

            print(f"{subtype_key!r} image Z={bestZ[i]} slice for {tested_keys_input[i]!r}")
            slice_2d = brain_voxels[:, :, bestZ[i], :].copy() if faYes else np.zeros((shapex, shapey, 3))
            # Add gamma corrected output to the green channel
            slice_2d[:, :, 1] += np.power(output_image[:, :, bestZ[i]], gamma)
            slice_2d = np.clip(slice_2d, 0, 1)
            matplotlib.pyplot.imshow(slice_2d, cmap="gray")
            # matplotlib.pyplot.colorbar()
            matplotlib.pyplot.show()

    if faYes:
        del fa_partition_voxels, fa_partition_lookup, fa_cloud_voxels, fa_cloud_lookup, brain_voxels
