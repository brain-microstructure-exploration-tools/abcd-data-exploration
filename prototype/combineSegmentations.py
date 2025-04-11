#!/usr/bin/env python3

"""
Prior to running this script you will probably have run something like

  transformSegmentationsToTemplate.sh

to create registered label files AF_left_Deformed.nii.gz, AF_right_Deformed.nii.gz, ATR_left_Deformed.nii.gz,
ATR_right_Deformed.nii.gz, CA_Deformed.nii.gz, CC_Deformed.nii.gz, CC_1_Deformed.nii.gz, CC_2_Deformed.nii.gz, ... for
each image.

As a first step, this script takes one brain region at a time and computes a consensus image from the individual image
descriptions of that region.  Instead of a value of 0 for background and 1 for foreground, a consensus image has a value
in the range [0, 1] at each voxel to represent the fraction of input images that had a value of 1 at that voxel.

As a second step, this script looks at all brain regions together and labels a voxel with the region with the highest
claim.  However, if no region assigns it a value >50% then the voxel is given the label of -1, for background.
"""

import glob
from typing import cast

import nibabel
import numpy as np
from numpy.typing import NDArray

data_dir: str = "/data/lee-data/abcd"
reference_image: str = data_dir + "/2024-11-15-gor/gortemplate0.nii.gz"
output_dir: str = data_dir + "/registration-experiments/2024-01"

deformed_string: str = "Deformed"
source_types: list[str] = ["mrtrix", "dipy"]
# Note that we have removed "CC" from `segmentations` because it is redundant with CC_1 through CC_7.
segmentations: list[str] = [
    "AF_left", "AF_right", "ATR_left", "ATR_right", "CA", "CC_1", "CC_2", "CC_3", "CC_4", "CC_5", "CC_6", "CC_7",
    "CG_left", "CG_right", "CST_left", "CST_right", "FPT_left", "FPT_right", "FX_left", "FX_right", "ICP_left",
    "ICP_right", "IFO_left", "IFO_right", "ILF_left", "ILF_right", "MCP", "MLF_left", "MLF_right", "OR_left",
    "OR_right", "POPT_left", "POPT_right", "SCP_left", "SCP_right", "SLF_III_left", "SLF_III_right", "SLF_II_left",
    "SLF_II_right", "SLF_I_left", "SLF_I_right", "STR_left", "STR_right", "ST_FO_left", "ST_FO_right", "ST_OCC_left",
    "ST_OCC_right", "ST_PAR_left", "ST_PAR_right", "ST_POSTC_left", "ST_POSTC_right", "ST_PREC_left", "ST_PREC_right",
    "ST_PREF_left", "ST_PREF_right", "ST_PREM_left", "ST_PREM_right", "T_OCC_left", "T_OCC_right", "T_PAR_left",
    "T_PAR_right", "T_POSTC_left", "T_POSTC_right", "T_PREC_left", "T_PREC_right", "T_PREF_left", "T_PREF_right",
    "T_PREM_left", "T_PREM_right", "UF_left", "UF_right",
]

source_type_list: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]] = []
for source_type in source_types:
    print(f"Processing {source_type}")
    mean_image_list: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    print("  Processing by segmentation")
    for segmentation in segmentations:
        print(f"    Processing {segmentation}")
        file_glob: str = (
            data_dir
            + "/registration-experiments/2024-01/tractseg_output_"
            + source_type
            + "/*_dwi/bundle_segmentations/"
            + segmentation
            + "_"
            + deformed_string
            + ".nii.gz"
        )
        file_list: list[str] = glob.glob(file_glob)
        all_images: NDArray[np.int32] = np.stack(
            [cast(nibabel.nifti1.Nifti1Image, nibabel.load(file)).get_fdata().astype(np.int32) for file in file_list],
            axis=0,
        )
        affine: NDArray[np.float64] = cast(nibabel.nifti1.Nifti1Image, nibabel.load(file_list[0])).affine
        mean_image: NDArray[np.float64] = np.mean(all_images, axis=0)
        mean_image_list.append((mean_image, affine))
        # all_values, all_counts = np.unique(all_images, return_counts=True)
        # typical_size: float = all_counts[1] / all_images.shape[0]
        # mean_values, mean_counts = np.unique(mean_image, return_counts=True)
        # cum_sum: NDArray[np.int32] = np.cumsum(mean_counts[::-1])[::-1]
        del file_glob, file_list, all_images, mean_image

    print("  Combining segmentations")
    source_type_image: NDArray[np.float64] = np.stack([e[0] for e in mean_image_list], axis=0)
    source_affine: NDArray[np.float64] = mean_image_list[0][1]
    source_img: nibabel.nifti1.Nifti1Image = nibabel.Nifti1Image(source_type_image, source_affine, dtype=np.float64)

    which_max_claim: NDArray[np.int32] = np.argmax(source_type_image, axis=0).astype(np.int32)
    max_claim: NDArray[np.float64] = np.amax(source_type_image, axis=0)
    threshold: float = 0.5
    which_max_claim[max_claim <= threshold] = -1
    which_img: nibabel.nifti1.Nifti1Image = nibabel.Nifti1Image(which_max_claim, source_affine, dtype=np.int32)

    print("  Writing two output files")
    source_filename: str = output_dir + "/segmentation_data_" + source_type + ".nii.gz"
    nibabel.save(source_img, source_filename)
    which_filename: str = output_dir + "/segmentation_labelmap_" + source_type + ".nii.gz"
    nibabel.save(which_img, which_filename)

    source_type_list.append((source_type_image, source_affine, which_max_claim))
    del mean_image_list, source_type_image, source_affine, source_img, source_filename
    del which_max_claim, max_claim, threshold, which_img, which_filename

print("source_type_list created")
