#!/usr/bin/env bash

# Prior to running this script you will probably have run something like
#
#   sh2peaks_batch.sh csd_output_dipy/   hdbet_output/ fod_peaks_dipy/
#   tractseg.sh       fod_peaks_dipy/    hdbet_output/ tractseg_output_dipy/
#   sh2peaks_batch.sh csd_output_mrtrix/ hdbet_output/ fod_peaks_mrtrix/
#   tractseg.sh       fod_peaks_mrtrix/  hdbet_output/ tractseg_output_mrtrix/
#
# to create label files AF_left.nii.gz, AF_right.nii.gz, ATR_left.nii.gz, ATR_right.nii.gz, CA.nii.gz, CC.nii.gz,
# CC_1.nii.gz, CC_2.nii.gz, ... for each image.
#
# This script takes these outputs and, using the same transforms that were used for their source whole images, registers
# each segmentation to the reference image.
#
# This script takes no arguments, but the variables that you are most likely to want to change might be these:

PATH="${HOME}/ants-2.5.3/bin:${PATH}"
data_dir="/data/lee-data/abcd"
reference_image="${data_dir}/2024-11-15-gor/gortemplate0.nii.gz"
deformed_string="Deformed"

for source_type in "mrtrix" "dipy"
do
    segmentations=$(ls $(\ls -d "${data_dir}"/registration-experiments/2024-01/tractseg_output_"${source_type}"/*_dwi | head -1)/bundle_segmentations | fgrep -v "${deformed_string}")
    for seg in ${segmentations}
    do
        echo "=== Processing ${seg%.nii.gz} ==="
        for moving_image in $(find "${data_dir}"/registration-experiments/2024-01/tractseg_output_"${source_type}"/*_dwi/bundle_segmentations -name "${seg}")
        do
            subject=$(basename "$(dirname "$(dirname "${moving_image}")")")
            warp=$(\ls "${data_dir}"/2024-11-15-gor/gorinput????-"${subject}"_fa-1Warp.nii.gz | head -1)
            generic_affine=$(\ls "${data_dir}"/2024-11-15-gor/gorinput????-"${subject}"_fa-0GenericAffine.mat | head -1)
            output_image="${moving_image%.nii.gz}_${deformed_string}.nii.gz"
            rm -f "${output_image}"
            antsApplyTransforms \
                -d 3 \
                -i "${moving_image}" \
                -r "${reference_image}" \
                -t "${warp}" \
                -t "${generic_affine}" \
                -o "${output_image}" \
                -n GenericLabel
        done
    done
done
