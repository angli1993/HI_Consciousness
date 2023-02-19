import numpy as np
import nibabel as nib
from nilearn import image, signal, plotting, masking
from scipy import stats
import seaborn as sns
import nilearn

## Taking HCP resting-state fMRI data as an example
## Volume pipeline
gradients_vol = image.load_img('$PATH_SOURCE/principal_gradient_MNI.nii.gz')
nii_img_demo = image.load_img('$HCP_PATH/3T/419239/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii.gz')
gradients_vol_res = image.resample_to_img(gradients_vol, nii_img_demo, interpolation='nearest')

gradients_vol_res_mask = image.binarize_img(gradients_vol_res)
gradients_vol_res_data = masking.apply_mask(gradients_vol_res, gradients_vol_res_mask)

s = ''
run = ''
test_nii_img = nib.load('$HCP_PATH/3T/' + s + '/MNINonLinear/Results/rfMRI_' + run + '/rfMRI_' + run + '_hp2000_clean.nii.gz')
test_nii_img_data = masking.apply_mask(test_nii_img, gradients_vol_res_mask)
test_nii_fdata = signal.clean(test_nii_img_data, detrend=False, standardize=False, 
                              confounds=None, low_pass=0.08, high_pass=0.01, t_r=0.72, ensure_finite=False)

HI_volume = stats.spearmanr(gradients_vol_res_data, np.std(test_nii_fdata, axis=0))[0]


## Surface pipeline
gradients = np.load('$PATH_SOURCE/embedding_dense_emb.npy')[:59412, 0]

s = ''
run = ''
test_img = nib.load('/share/data/dataset/hcpformat/3T/' + s + '/MNINonLinear/Results/rfMRI_' + run + '/rfMRI_' + run + '_Atlas_hp2000_clean.dtseries.nii')
test_data = np.array(test_img.dataobj)

test_fdata = nilearn.signal.clean(test_data, detrend=False, standardize=False, 
                                    confounds=None, low_pass=0.08, high_pass=0.01, t_r=0.72, ensure_finite=False)

HI_surface = stats.spearmanr(gradients, np.std(test_fdata[:, :59412], axis=0))[0]
