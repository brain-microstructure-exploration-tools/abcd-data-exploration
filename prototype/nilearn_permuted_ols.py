nilearn.mass_univariate.permuted_ols(tested_vars, target_vars, confounding_vars=None, model_intercept=True, n_perm=10000, two_sided_test=True, random_state=None, n_jobs=1, verbose=0, masker=None, tfce=False, threshold=None, output_type='legacy')

tested_vars.shape = (
    n_samples=number_of_images???,
    n_regressors=number_of_ksads_variables???
)
target_vars.shape = (
    n_samples=number_of_images,
    n_descriptors=number_of_voxels,
)

target_vars.shape = (n_samples=number_of_voxels, n_descriptors=number_of_images)
confounding_vars.shape = (n_samples=number_of_voxels, n_covars=???)
