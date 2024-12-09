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

import nilearn.datasets
import nilearn.glm
import nilearn.image
import nilearn.masking
import pandas as pd

# +
# Load functional data (replace with your own file path)
func_img = (
    "/data2/ABCD/gor-images/coregistered-images/"
    "gorinput0000-modality0-sub-NDARINV5U642ALM_ses-2YearFollowUpYArm1_run-concatenated_dwi_fa-WarpedToTemplate.nii.gz"
)

# Load or compute a brain mask
mask_img = nilearn.masking.compute_epi_mask(func_img)

# +
# TODO: Understand https://nilearn.github.io/dev/auto_examples/04_glm_first_level/plot_design_matrix.html
# Create a design matrix (replace with your own design matrix)
design_matrix = pd.DataFrame(...)

# Fit the GLM model
model = nilearn.glm.first_level.FirstLevelModel(mask_img=mask_img)
model.fit(func_img, design_matrix=design_matrix)

# +
# Define your contrasts
contrast_matrix = np.array([[1, 0, 0], [0, 1, 0]])  # Example contrast matrix

# Compute t-contrast maps
z_maps = model.compute_contrast(contrast_matrix, output_type="z_score")

# +
from nilearn.plotting import plot_stat_map

# Visualize the first contrast
plot_stat_map(
    z_maps[0], bg_img=nilearn.datasets.load_mni152_template(), title="t-contrast map for contrast 1"
)
