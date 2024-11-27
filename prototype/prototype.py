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

# Import from Python packages
from dipy.io.image import load_nifti, save_nifti
import csv
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
# Set global parameters to guide the workflow

# +
def get_list_of_image_files(directory):
    response = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".nii.gz")
    ]
    return response


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
        "subject",
        "session",
        "run",
        "image_type",
        "image_subtype",
        "processing",
    )
    response = [
        (file,)
        + (
            dict(
                zip(
                    filename_keys,
                    list(re.match(filename_pattern, os.path.basename(file)).groups()),
                )
            ),
        )
        + load_nifti(file, return_img=True)
        for file in list_of_files
    ]
    return response


def csv_file_to_raw_data(filename):
    """
    csv_file_to_raw_data returns a list of lists of str in row-major order
    """
    with open(filename, mode="r") as file:
        csv_reader = csv.reader(file)
        response = [row for row in csv_reader]
    return response


def csv_raw_to_row_dicts(csv_raw):
    """
    csv_raw_to_row_dicts converts a list of lists of str, in row-major order, to a list of dict of str.
    Each dict is formed by using the first row (column headers) as keys
    and a given data row as values of type str.
    """
    column_headers = csv_raw[0]
    response = [
        {column_headers[i]: row_value for i, row_value in enumerate(row)}
        for row in csv_raw[1:]
    ]
    return response


def csv_file_to_row_dicts(file):
    return csv_raw_to_row_dict(csv_file_to_raw_data(file))


def get_items(csv_row_dicts, key_dict):
    """
    get_items returns all rows (i.e., top-level elements) of csv_row_dicts, a list of dict of str,
    for which the row (a dict) contains all of the items in key_dict.
    """
    response = [
        row_dict
        for row_dict in csv_row_dicts
        if all(item in row_dict.items() for item in key_dict.items())
    ]
    return response


# +
list_of_image_files = random.sample(
    get_list_of_image_files(coregistered_images_directory), 10
)
image_data = get_data_from_image_files(list_of_image_files)

file_mh_y_ksads_ss = os.path.join(
    core_directory, "mental-health/mh_y_ksads_ss.csv"
)  # ksads_1_5_t (anhedonia), something with good entropy!!!
file_abcd_y_lt = os.path.join(
    core_directory, "abcd-general/abcd_y_lt.csv"
)  # site_id_l, rel_family_id, interview_age
file_abcd_p_demo = os.path.join(
    core_directory, "abcd-general/abcd_p_demo.csv"
)  # demo_gender_id_v2(_l)?
file_gish_p_gi = os.path.join(
    core_directory, "gender-identity-sexual-health/gish_p_gi.csv"
)  # demo_gender_id_v2(_l)?

csv_file = os.path.join(core_directory, "physical-health/ph_y_bld.csv")
csv_raw = csv_file_to_raw_data(csv_file)
row_dicts = csv_raw_to_row_dicts(csv_raw)
# -

if True:
    print(f"{len(image_data) = }")
    print(f"{type(image_data[0]) = }")
    print(f"{len(image_data[0]) = }")
    print("types(image_data[0]) = ", [type(e) for e in image_data[0]])
    print(f"{image_data[0][0] = }")
    print(f"{image_data[0][1] = }")
    print(f"{image_data[0][2].shape = }")
    print(f"{image_data[0][3].shape = }")
if True:
    print(f"{type(csv_raw) = }")
    print(f"{type(csv_raw[0]) = }")
    # print("csv_data row lengths = ", [len(row) for row in csv_raw])
    # print("Column 0 = ", [row[0] for row in csv_raw])
    print("Corner value = ", csv_raw[0][0])
if True:
    print(f"{type(row_dicts) = }")
    key0 = 0
    value0 = row_dicts[key0]
    key1 = next(iter(value0))
    value1 = value0[key1]
    print(f"row_dicts[{key0!r}][{key1!r}] = {value1!r}")
    # print(f"{row_dicts[0] = }")
    dict_of_keys = {key1: value1}
    print(f"{dict_of_keys = }")
    items = get_items(row_dicts, dict_of_keys)
    print(f"{len(items) = }")


