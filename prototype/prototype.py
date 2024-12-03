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
import random
import re
import os

# Set global parameters to match your environment
gor_image_directory = "/data2/ABCD/gor-images"
coregistered_images_directory = os.path.join(gor_image_directory, "coregistered-images")
tabular_data_directory = "/data2/ABCD/abcd-5.0-tabular-data-extracted"
core_directory = os.path.join(tabular_data_directory, "core")

# +
# The locations of some useful csv data columns

# These files live in `core_directory`

independent_vars = [
    [
        "abcd-general/abcd_y_lt.csv",
        [
            "site_id_l",  # Site ID at each event
            # TODO: We are including participants with no siblings in study at the expense of losing family ID.  Do this better.
            # "rel_family_id",  # Participants belonging to the same family share a family ID.  They will differ between data releases
            "interview_age",  # Participant's age in month at start of the event
        ],
    ],
    [
        "abcd-general/abcd_p_demo.csv",
        [
            "demo_gender_id_v2",  # 1=Male; 2=Female; 3=Trans male; 4=Trans female; 5=Gender queer; 6=Different; 777=Decline to answer; 999=Don't know
            # "demo_gender_id_v2_l",  # same?
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
# Functions for handling image voxel data


def get_list_of_image_files(directory):
    response = [
        os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".nii.gz")
    ]
    return response


def fix_filename_parsing(filename_parsing_dict):
    if "src_subject_id" in filename_parsing_dict.keys():
        filename_parsing_dict["src_subject_id"] = re.sub(
            r"^NDAR", "NDAR_", filename_parsing_dict["src_subject_id"]
        )
    eventname_conversion = {
        "baselineYear1Arm1": "baseline_year_1_arm_1",
        "1YearFollowUpYArm1": "1_year_follow_up_y_arm_1",
        "2YearFollowUpYArm1": "2_year_follow_up_y_arm_1",
        "3YearFollowUpYArm1": "3_year_follow_up_y_arm_1",
        "4YearFollowUpYArm1": "4_year_follow_up_y_arm_1",
    }
    if "eventname" in filename_parsing_dict.keys():
        filename_parsing_dict["eventname"] = eventname_conversion[
            filename_parsing_dict["eventname"]
        ]
    return filename_parsing_dict


def get_data_from_image_files(list_of_files):
    """
    get_data_from_image_files returns a list of tuples of 5 values:
    <class 'str'>: full file name
    <class 'dict'>: a parsing of the file name
    <class 'numpy.ndarray'>: image data as a numpy array
    <class 'numpy.ndarray'>: some other numpy array (transform?)
    <class 'nibabel.nifti1.Nifti1Image'>: an object that can be modified and written to file as a .nii.gz file
    """
    filename_pattern = r"gorinput([0-9]+)-modality([0-9]+)-sub-([A-Za-z0-9]+)_ses-([A-Za-z0-9]+)_run-([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)-([A-Za-z0-9]+).nii.gz"
    filename_keys = (
        "gorinput",
        "modality",
        "src_subject_id",
        "eventname",
        "run",
        "image_type",
        "image_subtype",
        "processing",
    )
    response = [
        (file,)
        + (
            fix_filename_parsing(
                dict(
                    zip(
                        filename_keys,
                        list(re.match(filename_pattern, os.path.basename(file)).groups()),
                    )
                )
            ),
        )
        + load_nifti(file, return_img=True)
        for file in list_of_files
    ]
    return response


def select_images(image_data, key_dict):
    # Currently we support only two keys
    assert all([key == "src_subject_id" or key == "eventname" for key, value in key_dict.items()])
    response = [
        image_row
        for image_row in image_data
        if all(item in image_row.items() for item in key_dict.items())
    ]
    return response

# +
# Functions for reading and selecting data from csv files


def _csv_file_to_list_of_lists(filename):
    """
    _csv_file_to_list_of_lists returns a list of lists of str in row-major order
    """
    with open(filename, mode="r") as file:
        csv_reader = csv.reader(file)
        response = [row for row in csv_reader]
    return response


def _list_of_lists_to_row_dicts(list_of_lists):
    """
    _list_of_lists_to_row_dicts converts a list of lists of str, in row-major order,
    to a list of dict of str.  Each dict is formed by using the first row (column headers)
    as keys and a given data row as values of type str.
    """
    column_headers = list_of_lists[0]
    response = [
        {column_headers[i]: row_value for i, row_value in enumerate(row)}
        for row in list_of_lists[1:]
    ]
    return response


def csv_file_to_row_dicts(file):
    return _list_of_lists_to_row_dicts(_csv_file_to_list_of_lists(file))


def select_items(list_of_row_dicts, key_dict):
    """
    select_items returns all rows (i.e., top-level elements) of list_of_row_dicts, a list of dict of str,
    for which the row (a dict) contains all of the items in key_dict.
    """
    response = [
        row_dict
        for row_dict in list_of_row_dicts
        if all(item in row_dict.items() for item in key_dict.items())
    ]
    return response

# +
# Functions for computing summary statistics for KSADS csv data


def row_dicts_to_counts_for_each_column(row_dicts):
    column_dict = {key: dict() for row in row_dicts for key in row.keys()}
    for row in row_dicts:
        for column_header, value in row.items():
            # 555 = "Not administered in the assessment",
            # so assume the same as "".  (TODO: Right?)
            value = "" if value == "555" else value
            # 888 = "Question not asked due to primary question response (branching logic)",
            # so assume it was obviously a "0".  (TODO: Right?)
            value = "0" if value == "888" else value
            column_dict[column_header][value] = column_dict[column_header].get(value, 0) + 1
    # Sort each value dict by its keys for easier reading
    column_dict = {
        key: dict(sorted(value.items(), key=lambda item: item[0], reverse=False))
        for key, value in column_dict.items()
    }
    return column_dict


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


def entropy_of_all_columns(all_columns):
    return {
        key: entropy_of_column_counts(value)
        for key, value in all_columns.items()
        if bool(re.match("ksads_\d", key))
    }


def find_interesting_entropies(file_mh_y_ksads_ss):
    # Find some KSADS data columns with high entropy.
    row_dicts_mh_y_ksads_ss = csv_file_to_row_dicts(file_mh_y_ksads_ss)
    print("Read done")

    counts_for_each_column_mh_y_ksads_ss = row_dicts_to_counts_for_each_column(
        row_dicts_mh_y_ksads_ss
    )
    print("Column counting done")

    entropies = entropy_of_all_columns(counts_for_each_column_mh_y_ksads_ss)
    sorted_entropies = dict(sorted(entropies.items(), key=lambda item: item[1], reverse=True))
    sorted_entropies = {
        key: (value, counts_for_each_column_mh_y_ksads_ss[key])
        for key, value in sorted_entropies.items()
        if bool(re.match("ksads_\d", key))
    }
    print("Entropy calculation done")
    return sorted_entropies


# +
# Find interesting KSADS data

file_mh_y_ksads_ss = "mental-health/mh_y_ksads_ss.csv"
if True:
    # User requests that we re-compute `interesting_ksads`
    if False:
        # User requests that we re-compute `sorted_entropies`
        try:
            del sorted_entropies
        except NameError:
            pass
    try:
        sorted_entropies
        print("Using cached value for sorted_entropies")
    except NameError:
        full_path = os.path.join(core_directory, file_mh_y_ksads_ss)
        # find_interesting_entropies is slow
        sorted_entropies = find_interesting_entropies(full_path)
    number_wanted = 20
    print(f"Distribution details: {list(sorted_entropies.items())[:number_wanted]}")
    interesting_ksads = list(sorted_entropies.keys())[:number_wanted]
    print(f"{interesting_ksads = }")
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

ksads_vars = [[file_mh_y_ksads_ss, interesting_ksads]]

# +
# Find all images for which we have tabular data

set_of_pairs_for_each_key = (
    {(row["src_subject_id"], row["eventname"]) for row in row_dict if row[key] != ""}
    for tablename, list_of_keys in independent_vars
    for row_dict in (csv_file_to_row_dicts(os.path.join(core_directory, tablename)),)
    for key in list_of_keys
)
set_intersection = functools.reduce(set.intersection, set_of_pairs_for_each_key)
print(f"{len(set_intersection) = }")
print(f"{set(list(set_intersection)[:20]) = }")
# TODO: If necessary, try fewer independent_vars so that we get more images
# -

# A random subset of all image files
list_of_image_files = random.sample(get_list_of_image_files(coregistered_images_directory), 10)
image_data = get_data_from_image_files(list_of_image_files)
print(f"file name parsing = {[image[1] for image in image_data]!r}")

if False:
    print(f"{len(image_data) = }")
    print(f"{type(image_data[0]) = }")
    print(f"{len(image_data[0]) = }")
    print("types(image_data[0]) = ", [type(e) for e in image_data[0]])
    print(f"{image_data[0][0] = }")
    print(f"{image_data[0][1] = }")
    print(f"{image_data[0][2].shape = }")
    print(f"{image_data[0][3].shape = }")
if False:
    print(f"{type(list_of_lists) = }")
    print(f"{type(list_of_lists[0]) = }")
    # print("csv_data row lengths = ", [len(row) for row in list_of_lists])
    # print("Column 0 = ", [row[0] for row in list_of_lists])
    print("Corner value = ", list_of_lists[0][0])
if False:
    print(f"{type(row_dicts) = }")
    key0 = 0
    value0 = row_dicts[key0]
    key1 = next(iter(value0))
    value1 = value0[key1]
    print(f"row_dicts[{key0!r}][{key1!r}] = {value1!r}")
    # print(f"{row_dicts[0] = }")
    dict_of_keys = {key1: value1}
    print(f"{dict_of_keys = }")
    items = select_items(row_dicts, dict_of_keys)
    print(f"{len(items) = }")


