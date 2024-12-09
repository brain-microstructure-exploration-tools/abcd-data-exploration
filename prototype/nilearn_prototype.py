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

# ## Installing dependencies
# One way to do this is with a virtual python environment installed so that it is accessible to Jupyter lab.  This needs to be set up only once.
# ```bash
# python -m venv ~/abcd311
# source ~/abcd311/bin/activate
# pip install ipykernel
# python -m ipykernel install --user --name abcd311
# pip install -r requirements.txt
# ```
# Then, once Jupyter is open with this lab notebook, "Change Kernel..." to be "abcd311".

# Import from Python packages
from dipy.io.image import load_nifti, save_nifti
import csv
import functools
import math
import numpy as np
import os
import pandas as pd
import random
import re
import statsmodels.api as sm
import time

# Set global parameters to match your environment
gor_image_directory = "/data2/ABCD/gor-images"
white_matter_mask_file = os.path.join(gor_image_directory, "gortemplate0.nii.gz")
coregistered_images_directory = os.path.join(gor_image_directory, "coregistered-images")
tabular_data_directory = "/data2/ABCD/abcd-5.0-tabular-data-extracted"
core_directory = os.path.join(tabular_data_directory, "core")

# +
# Useful inputs to our task

# independent_vars is the locations of some useful csv data columns. These files live in `core_directory`

independent_vars = [
    [
        "abcd-general/abcd_y_lt.csv",
        [
            # TODO: Convert site_id_l to one-hot so that we can use it.  Currently it is str
            # "site_id_l",  # Site ID at each event
            # TODO: We are including participants with no siblings in study at the expense of losing family ID.  Do this better.
            # "rel_family_id",  # Participants belonging to the same family share a family ID.  They will differ between data releases
            "interview_age"  # Participant's age in month at start of the event
        ],
    ],
    [
        "gender-identity-sexual-health/gish_p_gi.csv",
        [
            "demo_gender_id_v2",  # 1=Male; 2=Female; 3=Trans male; 4=Trans female; 5=Gender queer; 6=Different; 777=Decline to answer; 999=Don't know
            # "demo_gender_id_v2_l",  # same?
        ],
    ],
    [
        "abcd-general/abcd_p_demo.csv",
        [
            # TODO: These duplicate each other and gender-identity-sexual-health/gish_p_gi.csv.  Why?
            # "demo_gender_id_v2",  # same?
            # "demo_gender_id_v2_l",  # same?
        ],
    ],
    [
        "physical-health/ph_y_bld.csv",
        [
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
    ],
]

# We'll do KSADS variables once we've computed `interesting_ksads` below

# +
# Useful global variables

join_keys = ["src_subject_id", "eventname"]

# +
# Functions for handling image voxel data


def get_list_of_image_files(directory):
    pattern = r"^gorinput[0-9]{4}-.*\.nii\.gz$"
    response = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if bool(re.match(pattern, file))
    ]
    return response


def parse_image_filenames(list_of_image_files):
    """
    Returns a pandas DataFrame.
    The first column is the filename.  Additional columns indicate how the filename was parsed.
    For example, run as:
        df = parse_image_filenames(get_list_of_image_files(coregistered_images_directory))
    """
    filename_pattern = r"gorinput([0-9]+)-modality([0-9]+)-sub-([A-Za-z0-9]+)_ses-([A-Za-z0-9]+)_run-([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)-([A-Za-z0-9]+).nii.gz"
    filename_keys = [
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

    response = pd.DataFrame(
        [
            [filename, *list(re.match(filename_pattern, os.path.basename(filename)).groups())]
            for filename in list_of_image_files
        ],
        columns=filename_keys,
    )
    # Fix parsing of src_subject_id
    response["src_subject_id"] = [
        re.sub(r"^NDAR", "NDAR_", subject) for subject in response["src_subject_id"]
    ]
    # Fix parsing of eventname
    eventname_conversion = {
        "baselineYear1Arm1": "baseline_year_1_arm_1",
        "1YearFollowUpYArm1": "1_year_follow_up_y_arm_1",
        "2YearFollowUpYArm1": "2_year_follow_up_y_arm_1",
        "3YearFollowUpYArm1": "3_year_follow_up_y_arm_1",
        "4YearFollowUpYArm1": "4_year_follow_up_y_arm_1",
    }
    response["eventname"] = [eventname_conversion[event] for event in response["eventname"]]
    return response


def get_data_from_image_files(list_of_files):
    """
    get_data_from_image_files returns a list of tuples of 4 values:
    <class 'str'>: full file name
    <class 'numpy.ndarray'>: image data as a numpy array
    <class 'numpy.ndarray'>: some other numpy array (transform?)
    <class 'nibabel.nifti1.Nifti1Image'>: an object that can be modified and written to file as a .nii.gz file
    """
    response = [(file,) + load_nifti(file, return_img=True) for file in list_of_files]
    return response


def get_white_matter_mask(white_matter_mask_file):
    white_matter_mask_input = get_data_from_image_files([white_matter_mask_file])[0][1]
    mask_threshold = 0.70
    white_matter_mask = (white_matter_mask_input >= 0.70).reshape(-1)
    print(f"{np.sum(white_matter_mask) = }")
    return white_matter_mask

# +
# Functions for reading and selecting data from csv files


def csv_file_to_dataframe(filename):
    return pd.read_csv(filename)


# Select rows from data frame, to handle a common case where the direct pandas interface would be complicated
def select_rows_of_dataframe(df, query_dict):
    # Each key of query_dict is a column header of the df dataframe.
    # Each value of query_dict is a list of allowed values.
    # A row will be selected only if each of these columns has one of the allowed keys
    assert all(key in df.columns for key in query_dict.keys())
    # Old code: each of query_dict.values() is just a single value, not a list of values:
    # rows = df[
    #     functools.reduce(lambda x, y: x & y, [df[key] = value for key, value in query_dict.items()])
    # ]
    rows = df[
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
# Function to read and cache KSADS tabular information


def clean_ksads_data_frame(df):
    for column in df.columns:
        if bool(re.match("ksads_\d", column)):
            df.loc[df[column] == 555, column] = np.nan
            df.loc[df[column] == 888, column] = 0
    return df


def ksads_filename_to_dataframe(file_mh_y_ksads_ss, use_cache=True):
    rebuild_cache = not use_cache
    try:
        ksads_filename_to_dataframe.df_mh_y_ksads_ss
    except AttributeError:
        rebuild_cache = True
    if rebuild_cache:
        print("Begin reading KSADS data file")
        start = time.time()
        response = csv_file_to_dataframe(file_mh_y_ksads_ss)
        response = clean_ksads_data_frame(response)
        ksads_filename_to_dataframe.df_mh_y_ksads_ss = response
        print(f"Read KSADS data file in {time.time()-start}s")
    return ksads_filename_to_dataframe.df_mh_y_ksads_ss

# +
# Functions for computing summary statistics for KSADS csv data


def data_frame_value_counts(df):
    # Returns a dict:
    #     Each key is a column name
    #     Each value is a dict:
    #         Each key of this is a value that occurs in the column.
    #         The corresponding value is the number of occurrences.
    return {
        column: dict(df[column].value_counts(dropna=False).astype(int)) for column in df.columns
    }


def entropy_of_column_counts(column_counts):
    assert all(value >= 0 for value in column_counts.values())
    total_count = sum(column_counts.values())
    entropy = sum(
        [
            count / total_count * math.log2(total_count / count)
            for count in column_counts.values()
            if count > 0
        ]
    )
    return entropy


def ksads_keys_only(all_columns):
    return {key: value for key, value in all_columns.items() if bool(re.match("ksads_\d", key))}


def entropy_of_all_columns(all_columns):
    return {key: entropy_of_column_counts(value) for key, value in all_columns.items()}


def find_interesting_entropies(file_mh_y_ksads_ss):
    # Find some KSADS data columns with high entropy.
    df_mh_y_ksads_ss = ksads_filename_to_dataframe(file_mh_y_ksads_ss)
    counts_for_each_column_mh_y_ksads_ss = ksads_keys_only(
        data_frame_value_counts(df_mh_y_ksads_ss)
    )
    print("Column counting done")

    entropies = entropy_of_all_columns(counts_for_each_column_mh_y_ksads_ss)
    sorted_entropies = dict(sorted(entropies.items(), key=lambda item: item[1], reverse=True))
    sorted_entropies = {
        key: (value, counts_for_each_column_mh_y_ksads_ss[key])
        for key, value in sorted_entropies.items()
        # if bool(re.match("ksads_\d", key))
    }
    print("Entropy calculation done")
    return sorted_entropies

# +
# Find interesting KSADS data


def find_interesting_ksads():
    file_mh_y_ksads_ss = "mental-health/mh_y_ksads_ss.csv"
    if True:
        full_path = os.path.join(core_directory, file_mh_y_ksads_ss)
        # find_interesting_entropies is slow
        sorted_entropies = find_interesting_entropies(full_path)
        number_wanted = 3  # TODO: Pick a better number, like 20
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
    # print(f"{interesting_ksads = }")
    ksads_vars = [file_mh_y_ksads_ss, interesting_ksads]
    return ksads_vars

# +
# Find all images for which we have tabular data


def get_table_drop_nulls(tablename, list_of_keys):
    df = csv_file_to_dataframe(os.path.join(core_directory, tablename))[list_of_keys]
    df.replace("", pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df


def merge_dataframes_for_keys(independent_vars):
    df_generator = (
        get_table_drop_nulls(tablename, [*join_keys, *list_of_keys])
        for tablename, list_of_keys in independent_vars
        if list_of_keys
    )
    df_all_keys = next(df_generator)
    for df_next in df_generator:
        df_all_keys = pd.merge(
            df_all_keys, df_next, on=join_keys, how="inner", validate="one_to_one"
        )
    return df_all_keys


def merge_tabular_information(independent_vars, coregistered_images_directory):
    df_all_keys = merge_dataframes_for_keys(independent_vars)
    independent_keys = set(df_all_keys.columns).difference(set(join_keys))

    list_of_image_files = get_list_of_image_files(coregistered_images_directory)
    df_image_information = parse_image_filenames(list_of_image_files)
    df_all_images = pd.merge(
        df_all_keys,
        df_image_information[[*join_keys, "image_subtype", "filename"]],
        on=join_keys,
        how="inner",
        validate="one_to_many",
    )
    return df_all_images, independent_keys
# +
# Load tabular information for keys other than KSADS keys
start = time.time()
tabular_information, independent_keys = merge_tabular_information(
    independent_vars, coregistered_images_directory
)
print(f"{independent_keys = }")
print(f"{len(tabular_information) = }")
print(f"Total time to load tabular information = {time.time() - start}s")

# Load KSADS information
file_mh_y_ksads_ss, interesting_ksads = find_interesting_ksads()
df_mh_y_ksads_ss = ksads_filename_to_dataframe(file_mh_y_ksads_ss)

# Load voxel mask
white_matter_mask = get_white_matter_mask(white_matter_mask_file)


# -
def use_statsmodel():
    image_subtypes = list(tabular_information["image_subtype"].unique())
    an_image_filename = tabular_information["filename"].iloc[0]
    an_image_shape = get_data_from_image_files([an_image_filename])[0][1].shape

    all_subtypes = {}
    for image_subtype in image_subtypes:
        print(f"{image_subtype = }")
        subtype_information = tabular_information[
            tabular_information["image_subtype"] == image_subtype
        ]
        dict_of_images = {
            a: b
            for a, b, c, d in get_data_from_image_files(
                list(subtype_information["filename"].values)
            )
        }
        all_ksads_keys = {}
        for ksads_key in interesting_ksads:
            print(f"  {ksads_key = }")
            # Process only those images for which we have information for this ksads_key
            augmented_information = pd.merge(
                subtype_information,
                df_mh_y_ksads_ss[[*join_keys, ksads_key]],
                on=join_keys,
                how="inner",
                validate="one_to_one",
            )
            print(f"  {len(augmented_information) = }")
            augmented_information.dropna(inplace=True)
            print(f"  {len(augmented_information) = }")
            print(f"  {augmented_information.columns = }")
            # Now that we know which images we'll need, let's stack them into a single 4-dimensional shape
            all_images = np.stack(
                [dict_of_images[filename] for filename in augmented_information["filename"].values]
            )
            output_image = np.zeros(an_image_shape)
            for voxel_location, i in np.ndenumerate(output_image):
                # voxel_location = (28, 53, 71)  # TODO: Remove me
                # print(f"    {voxel_location = }")
                df_y = pd.DataFrame(all_images[:, *voxel_location], columns=["image"])
                if len(list(df_y["image"].unique())) <= 1:
                    response = 0.0  # Very bad voxel (despite being perfectly predictable)
                else:
                    y = df_y["image"]
                    X = augmented_information[[*independent_keys, ksads_key]]
                    X = sm.add_constant(X)  # TODO: Does this affect augmented_information?
                    # print(f"{type(y) = }")
                    # print(f"{type(X) = }")
                    # print(f"{df_y['image'].mean() = }")
                    # print(f"{df_y['image'].std() = }")
                    fit = sm.OLS(y, X).fit()
                    # print(fit.summary())
                    response = 1 - fit.f_pvalue  # Higher is better
                output_image[*voxel_location] = response
            all_ksads_keys[ksads_key] = output_image
        all_subtypes[image_subtype] = all_ksads_keys
    return all_subtypes
def use_numpy(white_matter_mask):
    image_subtypes = list(tabular_information["image_subtype"].unique())
    an_image_filename = tabular_information["filename"].iloc[0]
    an_image_shape = get_data_from_image_files([an_image_filename])[0][1].shape

    all_subtypes = {}
    for image_subtype in image_subtypes:
        print(f"{image_subtype = }")
        subtype_information = tabular_information[
            tabular_information["image_subtype"] == image_subtype
        ]
        dict_of_images = {
            a: b
            for a, b, c, d in get_data_from_image_files(
                list(subtype_information["filename"].values)
            )
        }
        all_ksads_keys = {}
        for ksads_key in interesting_ksads:
            print(f"  {ksads_key = }")
            # Process only those images for which we have information for this ksads_key
            augmented_information = pd.merge(
                subtype_information,
                df_mh_y_ksads_ss[[*join_keys, ksads_key]],
                on=join_keys,
                how="inner",
                validate="one_to_one",
            )
            augmented_information.dropna(inplace=True)
            augmented_information["constant"] = 1.0

            print(f"  {len(augmented_information) = }")
            augmented_information.dropna(inplace=True)
            print(f"  {len(augmented_information) = }")
            print(f"  {augmented_information.columns = }")

            X = augmented_information[[*independent_keys, ksads_key]].to_numpy()
            kernel = np.linalg.inv(X.transpose().dot(X))

            # Now that we know which images we'll need, let's stack them into a single 4-dimensional shape
            all_images = np.stack(
                [dict_of_images[filename] for filename in augmented_information["filename"].values]
            )
            y = all_images.reshape(all_images.shape[0], -1)[:, white_matter_mask]

            print(f"{X.shape = }")
            print(f"{kernel.shape = }")
            print(f"{y.shape = }")

            X_T_Y = X.transpose().dot(y)
            sum_of_squares = (y * y).sum(axis=0) - (X_T_Y * kernel.dot(X_T_Y)).sum(axis=0)
            print(f"{sum_of_squares.shape = }")
            output_image = np.zeros(white_matter_mask.shape)
            output_image[white_matter_mask] = -sum_of_squares
            output_image = output_image.reshape(all_images.shape[1:])

            all_ksads_keys[ksads_key] = output_image
        all_subtypes[image_subtype] = all_ksads_keys
    return all_subtypes


start = time.time()
subtype_ksadskey_image = use_numpy(white_matter_mask)
print(f"Computed all voxels in time {time.time() - start}s")


print(f"{type(subtype_ksadskey_image) = }")
print(f"{subtype_ksadskey_image.keys() = }")
print(f"{type(subtype_ksadskey_image['fa']) = }")
print(f"{subtype_ksadskey_image['fa'].keys() = }")
print(f"{type(subtype_ksadskey_image['fa']['ksads_1_187_t']) = }")
means = np.array(
    [
        [float(np.mean(value1.reshape(-1)[white_matter_mask])) for key1, value1 in value0.items()]
        for key0, value0 in subtype_ksadskey_image.items()
    ]
)
print(f"{means = }")
stds = np.array(
    [
        [float(np.std(value1.reshape(-1)[white_matter_mask])) for key1, value1 in value0.items()]
        for key0, value0 in subtype_ksadskey_image.items()
    ]
)
print(f"{stds = }")
print(f"Relative std = {stds/means!r}")


