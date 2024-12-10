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

# ## Installing dependent Python packages
# One way to do this is with a virtual python environment installed so that it is accessible to Jupyter lab.  This needs to be set up only once.
# ```bash
# python -m venv ~/abcd311
# source ~/abcd311/bin/activate
# pip install ipykernel
# python -m ipykernel install --user --name abcd311
# pip install -r requirements.txt
# ```
# Then, once Jupyter is open with this lab notebook, "Change Kernel..." to be "abcd311".

# ## Import dependent Pyton packages

# Import from Python packages
from typing import Any, Union
import csv
import dipy.io.image
import functools
import math
import nibabel.nifti1
import nilearn.masking
import nilearn.mass_univariate
import numpy as np
import os
import pandas as pd
import random
import re
import statsmodels.api as sm
import time

# ## Set global variables

# +
# Set global parameters to match your environment.  Ultimately these will be member variables of a class
gor_image_directory: str = "/data2/ABCD/gor-images"
white_matter_mask_file: str = os.path.join(gor_image_directory, "gortemplate0.nii.gz")
coregistered_images_directory: str = os.path.join(gor_image_directory, "coregistered-images")
tabular_data_directory: str = "/data2/ABCD/abcd-5.0-tabular-data-extracted"
core_directory: str = os.path.join(tabular_data_directory, "core")

# Useful class static const member
# Images are distinguished from each other by their subjects and timing.  (Also distinguished as "md" vs. "fa" type, though not relevant here.)
join_keys: list[str] = ["src_subject_id", "eventname"]
# -

# ## Define functions for various steps of the workflow

# +
# Functions for handling image voxel data


def get_list_of_image_files(directory: str) -> list[str]:
    """
    Returns a list of full path names of image files.  Input is the directory in which to look for these files.
    """
    # Choose the pattern to get .nii.gz files but avoid the template files such as gortemplate0.nii.gz
    pattern: str = r"^gorinput[0-9]{4}-.*\.nii\.gz$"
    response: list[str] = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if bool(re.match(pattern, file))
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
        r"gorinput([0-9]+)-modality([0-9]+)-sub-([A-Za-z0-9]+)_ses-([A-Za-z0-9]+)_run-([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)-([A-Za-z0-9]+).nii.gz"
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
            [filename, *list(re.match(filename_pattern, os.path.basename(filename)).groups())]
            for filename in list_of_image_files
        ],
        columns=filename_keys,
    )

    # Fix parsing of src_subject_id.  The filenames do not have an underscore after "NDAR",
    # but src_subject_id values in the data tables do.
    response["src_subject_id"] = [
        re.sub(r"^NDAR", "NDAR_", subject) for subject in response["src_subject_id"]
    ]
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
        <class 'numpy.ndarray'>: some other numpy array (transform?)
        <class 'nibabel.nifti1.Nifti1Image'>: an object that can be modified and can be written to file as a .nii.gz file
    """
    response = [(file,) + dipy.io.image.load_nifti(file, return_img=True) for file in list_of_files]
    return response


def get_white_matter_mask_as_numpy(
    white_matter_mask_file: str, mask_threshold: float
) -> np.ndarray:
    # The return value is a np.ndarray of bool, a voxel mask that indicates which voxels are to be kept for subsequent analyses.
    # The mask is flattened to a single dimension because our analysis software indexes voxels this way
    # We determine white matter by looking at the white_matter_mask_file for voxels that have an intensity above a threshold
    white_matter_mask_input: np.ndarray = get_data_from_image_files([white_matter_mask_file])[0][1]
    white_matter_mask: np.ndarray = (white_matter_mask_input >= mask_threshold).reshape(-1)
    print(f"Number of white matter voxels = {np.sum(white_matter_mask)}")
    return white_matter_mask

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
    # Each key of query_dict is a column header of the df dataframe.
    # Each value of query_dict is a list of allowed values.
    # A row will be selected only if each of these columns has one of the allowed keys
    assert all(key in df.columns for key in query_dict.keys())
    # Old code: each of query_dict.values() is just a single value, not a list of values:
    # rows = df[
    #     functools.reduce(lambda x, y: x & y, [df[key] = value for key, value in query_dict.items()])
    # ]
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
        that 888 means that the question was not asked because the answer was already known to be "absent".
        that 555 means we don't know the answer.
    See https://wiki.abcdstudy.org/release-notes/non-imaging/mental-health.html
    """
    for column in df.columns:
        if bool(re.match("ksads_\d", column)):
            df[column] = df[column].astype(float)
            df.loc[df[column] == 555, column] = np.nan
            df.loc[df[column] == 888, column] = 0
    return df


def ksads_filename_to_dataframe(
    file_mh_y_ksads_ss: str, use_cache: bool = True
) -> pd.core.frame.DataFrame:
    """
    Read in the KSADS data, or grab it from cache if the user requests it and we have it.
    """
    rebuild_cache: bool = not use_cache
    try:
        # Check whether we've cached this computation
        ksads_filename_to_dataframe.df_mh_y_ksads_ss
    except AttributeError:
        # We haven't cached it, so we'll have to read the data even if the user would have permitted us to use the cache
        rebuild_cache = True
    if rebuild_cache:
        print("Begin reading KSADS data file")
        start: float = time.time()
        # Reading from disk takes 10-30 seconds; it is a big file
        response: pd.core.frame.DataFrame = csv_file_to_dataframe(file_mh_y_ksads_ss)
        # Deal with 555 and 888 values in the data table
        response = clean_ksads_data_frame(response)
        # Place the computed table in the cache
        ksads_filename_to_dataframe.df_mh_y_ksads_ss = response
        print(f"Read KSADS data file in {time.time()-start}s")
    # Return what is now in the cache
    return ksads_filename_to_dataframe.df_mh_y_ksads_ss

# +
# Functions for computing summary statistics for KSADS csv data


def data_frame_value_counts(df: pd.core.frame.DataFrame) -> dict[str, dict[str, np.int64]]:
    """
    Whether an KSADS column of data is interesting depends upon, in part, how much it varies across images.
    Here we performa census that counts how many times each value occurs in a column.
    """
    # Returns a dict:
    #     Each key is a column name
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
        [
            count / total_count * math.log2(total_count / count)
            for count in column_counts.values()
            if count > 0
        ]
    )
    return entropy


def ksads_keys_only(all_columns: dict[str, Any]) -> dict[str, Any]:
    """
    This finds those keys that are KSADS variables rather than being index keys (subject, event), etc.
    """
    return {key: value for key, value in all_columns.items() if bool(re.match("ksads_\d", key))}


def entropy_of_all_columns(all_columns) -> dict[str, float]:
    """
    Compute entropy (information) of every column by calling subroutine for each column
    """
    return {key: entropy_of_column_counts(value) for key, value in all_columns.items()}


def find_interesting_entropies(file_mh_y_ksads_ss: str):
    """
    Compute the entropy (information) of each KSADS variable and return them sorted from most entropy to least
    """
    # Find some KSADS data columns with high entropy.
    df_mh_y_ksads_ss: pd.core.frame.DataFrame = ksads_filename_to_dataframe(file_mh_y_ksads_ss)
    counts_for_each_column_mh_y_ksads_ss: dict[str, Any] = ksads_keys_only(
        data_frame_value_counts(df_mh_y_ksads_ss)
    )
    print("Column counting done")

    entropies: dict[str, float] = entropy_of_all_columns(counts_for_each_column_mh_y_ksads_ss)
    sorted_entropies: dict[str, Any] = dict(
        sorted(entropies.items(), key=lambda item: item[1], reverse=True)
    )
    sorted_entropies = {
        key: (value, counts_for_each_column_mh_y_ksads_ss[key])
        for key, value in sorted_entropies.items()
        # if bool(re.match("ksads_\d", key))
    }
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
    file_mh_y_ksads_ss: str = "mental-health/mh_y_ksads_ss.csv"
    if True:
        full_path = os.path.join(core_directory, file_mh_y_ksads_ss)
        sorted_entropies = find_interesting_entropies(full_path)
        number_wanted = 4  # TODO: Is 20 a good number?
        interesting_ksads = list(sorted_entropies.keys())[:number_wanted]
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
    ksads_vars: tuple[str, list[str]] = (file_mh_y_ksads_ss, interesting_ksads)
    return ksads_vars

# +
# Find all images for which we have tabular data


def get_table_drop_nulls(tablename: str, list_of_keys: list[str]) -> pd.core.frame.DataFrame:
    """
    Reads data table from filename
    Keeps only specified keys (columns)
    Replace each empty string with a NaN value
    Converts all columns (except those in join_keys) to float
    Deletes rows that include NaN values.
    """
    df: pd.core.frame.DataFrame = csv_file_to_dataframe(os.path.join(core_directory, tablename))[
        list_of_keys
    ]
    df.replace("", pd.NA, inplace=True)
    for col in list_of_keys:
        if col not in join_keys:
            df[col] = df[col].astype(float)
    df.dropna(inplace=True)
    return df


def merge_dataframes_for_keys(
    confounding_vars: list[tuple[str, list[str]]]
) -> pd.core.frame.DataFrame:
    """
    This routine merges data tables for the confounding variables into a single table.

    For each data table and its list of keys:
        get it as a dataframe
    Merge these dataframes into a single table using the join_keys
    """
    df_generator: pd.core.frame.DataFrame = (
        get_table_drop_nulls(tablename, [*join_keys, *list_of_keys])
        for tablename, list_of_keys in confounding_vars
        if list_of_keys
    )
    df_all_keys: pd.core.frame.DataFrame = next(df_generator)
    for df_next in df_generator:
        df_all_keys = pd.merge(
            df_all_keys, df_next, on=join_keys, how="inner", validate="one_to_one"
        )
    return df_all_keys


def merge_confounding_table(
    confounding_vars: list[tuple[str, list[str]]], coregistered_images_directory: str
) -> tuple[pd.core.frame.DataFrame, list[str]]:
    """
    This creates the master data table from disparate sources that includes confounding variables and image meta data.
    It also returns a list of the confounding variable names

    Create a dataframe that contains all confounding variables (regardless of their source data table)
    Create a dataframe that contains meta data about the available images
    Merge these two dataframes using the join_keys
    Note that a (src_subject_id, eventname) pair can occur more than once, e.g., for both "md" and "fa" image subtypes
    """
    df_all_keys: pd.core.frame.DataFrame = merge_dataframes_for_keys(confounding_vars)
    confounding_keys: list[str] = list(set(df_all_keys.columns).difference(set(join_keys)))

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


# -
# TODO: describe this function
# How we might process the inputs using numpy
def use_numpy(
    white_matter_mask: np.ndarray,
    confounding_table: pd.core.frame.DataFrame,
    interesting_ksads: list[str],
    tested_vars: pd.core.frame.DataFrame,
    confounding_keys: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    image_subtypes: list[str] = list(confounding_table["image_subtype"].unique())
    # an_image_filename = confounding_table["filename"].iloc[0]
    # an_image_shape = get_data_from_image_files([an_image_filename])[0][1].shape

    all_subtypes: dict[str, dict[str, np.ndarray]] = {}
    for image_subtype in image_subtypes:
        print(f"  {image_subtype = }")
        subtype_information: pd.core.frame.DataFrame = confounding_table[
            confounding_table["image_subtype"] == image_subtype
        ]
        dict_of_images: dict[str, np.ndarray] = {
            a: b
            for a, b, c, d in get_data_from_image_files(
                list(subtype_information["filename"].values)
            )
        }
        all_ksads_keys: dict[str, np.ndarray] = {}
        for ksads_key in interesting_ksads:
            print(f"    {ksads_key = }")
            # Process only those images for which we have information for this ksads_key
            augmented_information: pd.core.frame.DataFrame = pd.merge(
                subtype_information,
                tested_vars[[*join_keys, ksads_key]],
                on=join_keys,
                how="inner",
                validate="one_to_one",
            )
            augmented_information.dropna(inplace=True)
            augmented_information["constant"] = 1.0

            augmented_information.dropna(inplace=True)
            print(f"    {augmented_information.columns = }")
            print(f"    {len(augmented_information) = }")

            X: np.ndarray = augmented_information[[*confounding_keys, ksads_key]].to_numpy()
            kernel: np.ndarray = np.linalg.inv(X.transpose().dot(X))

            # Now that we know which images we'll need, let's stack them into a single 4-dimensional shape
            all_images: np.ndarray = np.stack(
                [dict_of_images[filename] for filename in augmented_information["filename"].values]
            )
            y: np.ndarray = all_images.reshape(all_images.shape[0], -1)[:, white_matter_mask]

            print(f"    {X.shape = }")
            print(f"    {kernel.shape = }")
            print(f"    {y.shape = }")

            X_T_Y: np.ndarray = X.transpose().dot(y)
            sum_of_squares: np.ndarray = (y * y).sum(axis=0) - (X_T_Y * kernel.dot(X_T_Y)).sum(
                axis=0
            )
            print(f"    {sum_of_squares.shape = }")
            # TODO: Do we need to make sure that this background (zeros) is worse than foreground?
            output_image: np.ndarray = np.zeros(white_matter_mask.shape)
            output_image[white_matter_mask] = -sum_of_squares
            output_image = output_image.reshape(all_images.shape[1:])

            all_ksads_keys[ksads_key] = output_image
        all_subtypes[image_subtype] = all_ksads_keys
    return all_subtypes


# TODO: describe this function
# How we might process the inputs using numpy
def use_nilearn(
    white_matter_mask: nibabel.nifti1.Nifti1Image,
    confounding_table: pd.core.frame.DataFrame,
    interesting_ksads: list[str],
    tested_vars: pd.core.frame.DataFrame,
    confounding_keys: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    # print(f"{white_matter_mask = }")
    # # confounding_table has columns *counfounding_vars, src_subject_id, eventname, image_subtype, filename
    # print(f"{confounding_table = }")
    # print(f"{interesting_ksads = }")  # list of str of selected ksads
    # print(f"{tested_vars = }")  # Dataframe across all ksads
    # print(f"{confounding_keys = }")  # list of str of confounding keys

    image_subtypes = list(confounding_table["image_subtype"].unique())
    # an_image_filename = confounding_table["filename"].iloc[0]
    # an_image_shape = get_data_from_image_files([an_image_filename])[0][1].shape

    all_subtypes = {}
    for image_subtype in image_subtypes:
        print(f"{image_subtype = }")

        all_information = pd.merge(
            confounding_table[confounding_table["image_subtype"] == image_subtype],
            tested_vars[[*join_keys, *interesting_ksads]],
            on=join_keys,
            how="inner",
            validate="one_to_one",
        )
        all_information.dropna(inplace=True)

        """
        Although pandas is a great way to read and manipulate these data tables, nilearn expects them to be "array-like".
        Because panda tables require `.iloc` to accept integer coordinates for a 2d-table, panda tables fail "array-like".
        We'll use numpy arrays.
        """
        # tested_input: pd.core.frame.DataFrame = all_information[interesting_ksads]
        tested_input: np.ndarray = all_information[interesting_ksads].to_numpy(dtype=float)
        print(f"  {tested_input.shape = }")
        # confounding_input: pd.core.frame.DataFrame = all_information[confounding_keys]
        confounding_input: np.ndarray = all_information[confounding_keys].to_numpy(dtype=float)
        print(f"  {confounding_input.shape = }")
        # target_input: pd.core.frame.DataFrame = pd.DataFrame(
        #     [
        #         np_voxels.reshape(-1)[white_matter_mask]
        #         for a, np_voxels, c, d in get_data_from_image_files(
        #             list(all_information["filename"].values)
        #         )
        #     ]
        # )
        target_input: np.ndarray = np.stack(
            [
                np_voxels.reshape(-1)
                for a, np_voxels, c, d in get_data_from_image_files(
                    list(all_information["filename"].values)
                )
            ]
        )
        print(f"  {target_input.shape = }")

        model_intercept: bool = True
        n_perm: int = 100  # TODO: Use 10000
        two_sided_test: bool = True
        random_state = None
        n_jobs: int = 1  # TODO: Use -1
        verbose: int = 1
        masker = white_matter_mask
        tfce: bool = False  # TODO: Set to True
        threshold = None
        output_type: str = "dict"
        response: dict[str, np.ndarray] = nilearn.mass_univariate.permuted_ols(
            tested_vars=tested_input,
            target_vars=target_input,
            confounding_vars=confounding_input,
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
        all_subtypes[image_subtype] = response
    return all_subtypes


# ## Define or load input data

# +
# Set inputs for our task.  Ultimately these will either be class members or parameters for class methods.

# confounding_vars is the locations of some useful csv data columns. These files live in `core_directory`
confounding_vars_input: list[tuple[str, list[str]]] = [
    (
        "abcd-general/abcd_y_lt.csv",
        [
            # TODO: Convert site_id_l, modified_rel_family_id, and demo_gender_id_v2 to one-hot so that we can use them.
            # Currently they are str, Optional[int], and int.
            # "site_id_l",  # Site ID at each event
            # TODO: We are including participants with no siblings in study at the expense of losing family ID.  Do this better.
            # "rel_family_id",  # Participants belonging to the same family share a family ID.  They will differ between data releases
            "interview_age"  # Participant's age in month at start of the event
        ],
    ),
    (
        "gender-identity-sexual-health/gish_p_gi.csv",
        [
            # TODO: demo_gender_id_v2 should be one-hot, with handling for 777 and 999 as category="unknown"?
            "demo_gender_id_v2",  # 1=Male; 2=Female; 3=Trans male; 4=Trans female; 5=Gender queer; 6=Different; 777=Decline to answer; 999=Don't know
            # "demo_gender_id_v2_l",  # same?, so don't include it
        ],
    ),
    (
        "abcd-general/abcd_p_demo.csv",
        [
            # TODO: These duplicate each other and gender-identity-sexual-health/gish_p_gi.csv.  Why?
            # "demo_gender_id_v2",  # same?, so don't include it
            # "demo_gender_id_v2_l",  # same?, so don't include it
        ],
    ),
    (
        "physical-health/ph_y_bld.csv",
        [
            # TODO: Can any of these be useful, or are they sparse/empty of useful information?
            # "biospec_blood_baso_percent",  # BASO %
            # "biospec_blood_baso_abs",  # BASO ABS
            # "biospec_blood_eos_percent",  # EOS %
            # "biospec_blood_eos_abs",  # EOS ABS
            # "biospec_blood_hemoglobin",  # Hemoglobin
            # "biospec_blood_mcv",  # MCV
            # "biospec_blood_plt_count",  # PLT Count
            # "biospec_blood_wbc_count",  # WBC Count
            # "biospec_blood_ferritin",  # Ferritin
            # "biospec_blood_hemoglobin_a1",  # hemoglobin_a1
            # "biospec_blood_imm_gran_per",  # Immature Gran %
        ],
    ),
]
# +
# Load tabular information for counfounding variables
# Merge the information with image meta data so that we know which images we have confounding variable values for.
start = time.time()
confounding_table_input, confounding_keys_input = merge_confounding_table(
    confounding_vars_input, coregistered_images_directory
)
print(f"{confounding_keys_input = }")
print(f"{len(confounding_table_input) = }")
print(f"Total time to load confounding information = {time.time() - start}s")

# Load KSADS information
file_mh_y_ksads_ss_input, interesting_ksads_input = find_interesting_ksads()
tested_vars_input = ksads_filename_to_dataframe(file_mh_y_ksads_ss_input)
# -
# ## Run the workflow


# +
mask_threshold: float = 0.70
if False:
    print("Invoking use_numpy")
    white_matter_mask_input = get_white_matter_mask_as_numpy(white_matter_mask_file, mask_threshold)
    func = use_numpy
else:
    print("Invoking use_nilearn")
    white_matter_mask_input = nilearn.masking.compute_brain_mask(
        target_img=white_matter_mask_file,
        threshold=mask_threshold,
        connected=False,  # TODO: Is this best?
        opening=False,  # False or positive int
        memory=None,
        verbose=2,
        mask_type="whole-brain",  # "whole-brain", "gm", "wm"
    )
    print(f"Number of white_matter voxels = {np.sum(white_matter_mask_input.get_fdata())}")
    print(
        f"Total number of voxels = {np.sum((white_matter_mask_input.get_fdata() == 0.0) | (white_matter_mask_input.get_fdata() == 1.0))}"
    )
    print(
        f"zero = {np.sum((white_matter_mask_input.get_fdata() != 0.0) & (white_matter_mask_input.get_fdata() != 1.0))}"
    )
    func = use_nilearn

start = time.time()
output_images_by_subtype: dict[str, dict[str, np.ndarray]] = func(
    white_matter_mask_input,
    confounding_table_input,
    interesting_ksads_input,
    tested_vars_input,
    confounding_keys_input,
)
print(f"Computed all voxels in time {time.time() - start}s")
# -


# ## Show some output

print(f"{list(output_images_by_subtype.keys()) = }")
print(f"{list(output_images_by_subtype['fa'].keys()) = }")

if func == use_numpy:
    print("## use_numpy output")
    print(f"{type(output_images_by_subtype['fa']['ksads_1_187_t']) = }")
    means = np.array(
        [
            [
                float(np.mean(value1.reshape(-1)[white_matter_mask_input]))
                for key1, value1 in value0.items()
            ]
            for key0, value0 in output_images_by_subtype.items()
        ]
    )
    print(f"{means = }")
    stds = np.array(
        [
            [
                float(np.std(value1.reshape(-1)[white_matter_mask_input]))
                for key1, value1 in value0.items()
            ]
            for key0, value0 in output_images_by_subtype.items()
        ]
    )
    print(f"{stds = }")
    print(f"Relative std = {stds/means!r}")
else:
    print("Skipped use_numpy output")

if func == use_nilearn:
    print("## use_nilearn output")
    for sub_type in ("fa", "md"):
        print(f"## {sub_type = }")
        for table in output_images_by_subtype[sub_type].keys():
            print(f"## table: {table = }")
            print(f"{output_images_by_subtype[sub_type][table].shape = }")
            print(f"{np.sum(output_images_by_subtype[sub_type][table]) = }")
            print(f"{np.sum(~np.isnan(output_images_by_subtype[sub_type][table])) = }")
            # print(f"{output_images_by_subtype[sub_type]['t'] = }")
            # print(f"{output_images_by_subtype[sub_type]['logp_max_t'] = }")
            # print(f"{output_images_by_subtype[sub_type]['h0_max_t'] = }")
else:
    print("Skipped use_nilearn output")


