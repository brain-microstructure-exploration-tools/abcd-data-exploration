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
from typing import Any

import numpy as np
import packaging.version
import slicerio
import slicerio._version
from numpy.typing import NDArray

data_dir: str = "/data/lee-data/abcd"
reference_filename: str = data_dir + "/2024-11-15-gor/gortemplate0.nii.gz"
output_dir: str = data_dir + "/registration-experiments/2024-01"

deformed_string: str = "Deformed"
source_types: list[str] = ["mrtrix", "dipy"]
# Note that we have removed "CC" from `labels` because it is redundant with CC_1 through CC_7.
labels: list[str] = [
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
segment_descriptions: list[dict[str, Any]] = [
    dict(id=f"Segment_{i}", labelValue=i, name=label) for i, label in enumerate(labels)
]

output_of_source_types: list[tuple[dict[str, Any], dict[str, Any]]] = []
for source_type in source_types:
    print(f"Processing {source_type}", flush=True)
    mean_arrays_of_segments: list[tuple[NDArray[np.float64], dict[str, Any]]] = []
    print("  Processing by segmentation", flush=True)
    for label in labels:
        print(f"    Processing {label}", flush=True)
        file_glob: str = (
            data_dir
            + "/registration-experiments/2024-01/tractseg_output_"
            + source_type
            + "/*_dwi/bundle_segmentations/"
            + label
            + "_"
            + deformed_string
            + ".nii.gz"
        )
        filenames_of_subjects: list[str] = glob.glob(file_glob)
        all_input_arrays: NDArray[np.int32] = np.stack(
            [slicerio.read_segmentation(file)["voxels"].astype(np.int32) for file in filenames_of_subjects], axis=0
        )
        # Note, first two rows of affine are sign flipped relative to nibabel.load(filenames_of_subjects[0]).affine
        header: dict[str, Any] = {**slicerio.read_segmentation(filenames_of_subjects[0])}
        del header["voxels"]
        mean_array: NDArray[np.float64] = np.mean(all_input_arrays, axis=0)
        mean_arrays_of_segments.append((mean_array, header))
        del file_glob, filenames_of_subjects, all_input_arrays, header, mean_array

    print("  Combining segmentations", flush=True)
    source_img_voxels = np.stack([e[0] for e in mean_arrays_of_segments], axis=0).astype(np.float64)
    argmax_claim: NDArray[np.int32] = np.argmax(source_img_voxels, axis=0).astype(np.int32)
    max_claim: NDArray[np.float64] = np.amax(source_img_voxels, axis=0)
    threshold: float = 0.5
    argmax_claim[max_claim <= threshold] = -1

    print("  Writing output files", flush=True)
    source_img: dict[str, Any] = {**mean_arrays_of_segments[0][1]}
    source_img["voxels"] = source_img_voxels
    argmax_img: dict[str, Any] = {**mean_arrays_of_segments[0][1]}
    argmax_img["voxels"] = argmax_claim
    argmax_img["segments"] = segment_descriptions
    if packaging.version.Version(slicerio._version.__version__) > packaging.version.Version("1.1.2"):
        # This is a 4-dimensional array (label, x, y, z).  Slicerio 1.1.2 chokes in this case, but slicerio
        # 378205b394a0db277041cfd6f1559542b026fce2 or newer works.
        source_filename: str = output_dir + "/segmentation_data_" + source_type + ".seg.nrrd"
        slicerio.write_segmentation(source_filename, source_img)

    argmax_filename: str = output_dir + "/segmentation_labelmap_" + source_type + ".seg.nrrd"
    slicerio.write_segmentation(argmax_filename, argmax_img)

    output_of_source_types.append((source_img, argmax_img))
    del mean_arrays_of_segments, source_img_voxels, argmax_claim, max_claim, threshold, source_img, argmax_img

print(f"output_of_source_types created.  Also see {output_dir}/.", flush=True)
