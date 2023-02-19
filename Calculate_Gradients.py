## Reference from https://github.com/NeuroanatomyAndConnectivity/gradient_analysis/blob/master/01_create_human_connectome.ipynb
## https://github.com/satra/mapalign
## Download MSM-All-registered group-average rfMRI dense connectivity data: HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii

import sys
sys.path.append("$PATH/mapalign-master")
from mapalign import embed
import numpy as np
import nibabel as nib 
from sklearn.metrics import pairwise_distances

dcon = np.tanh(nib.load('$PATH/HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii').dataobj)

# Get number of nodes
N = dcon.shape[0]

# Generate percentile thresholds for 90th percentile
perc = np.array([np.percentile(x, 90) for x in dcon])

# Threshold each row of the matrix by setting values below 90th percentile to 0
for i in range(dcon.shape[0]):
    # print("Row %d" % i)
    dcon[i, dcon[i,:] < perc[i]] = 0


# Check for minimum value
print("Minimum value is %f" % dcon.min())

# The negative values are very small, but we need to know how many nodes have negative values
# Count negative values per row
neg_values = np.array([sum(dcon[i,:] < 0) for i in range(N)])
print("Negative values occur in %d rows" % np.sum(neg_values > 0))

# Since there are only 23 vertices with total of 5000 very small negative values, we set these to zero
dcon[dcon < 0] = 0

aff = 1 - pairwise_distances(dcon, metric = 'cosine')

emb = embed.compute_diffusion_map(aff, alpha = 0.5, return_result=True)


np.save(emb[0], '$PATH/embedding_dense_emb.npy')

reference = '$PATH_SOURCE/demo.dscalar.nii'
demo = nib.load(reference)
demo_new = nib.cifti2.cifti2.Cifti2Image(dataobj=emb[:, 0].reshape(1, 91282), header=demo.header, 
                                         nifti_header=demo.nifti_header, extra=demo.extra, file_map=demo.file_map)

demo_new.to_filename('$PATH_SOURCE/hcp.embed.1.dscalar.nii')

# HCP template available from https://balsa.wustl.edu/reference/pkXDZ

# wb_command -cifti-separate $PATH_SOURCE/hcp.embed.1.dscalar.nii COLUMN \
#     -metric CORTEX_LEFT $PATH_SOURCE/hcp.embed.1.L.metric \
#     -metric CORTEX_RIGHT $PATH_SOURCE/hcp.embed.1.R.metric \
#     -volume-all $PATH_SOURCE/hcp.embed.1.volume.nii
  
# wb_command -metric-to-volume-mapping $PATH_SOURCE/hcp.embed.1.L.metric \
#     ~/templates/HCP_S1200_Atlas_Z4_pkXDZ/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii \
#     /localdata/software/FSL/data/standard/MNI152_T1_2mm_brain.nii.gz $PATH_SOURCE/volume.1.L.nii \
#     -ribbon-constrained ~/templates/HCP_S1200_Atlas_Z4_pkXDZ/S1200.L.white_MSMAll.32k_fs_LR.surf.gii \
#     ~/templates/HCP_S1200_Atlas_Z4_pkXDZ/S1200.L.pial_MSMAll.32k_fs_LR.surf.gii -greedy

# wb_command -metric-to-volume-mapping $PATH_SOURCE/hcp.embed.1.R.metric \
#     ~/templates/HCP_S1200_Atlas_Z4_pkXDZ/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii \
#     /localdata/software/FSL/data/standard/MNI152_T1_2mm_brain.nii.gz $PATH_SOURCE/volume.1.R.nii \
#     -ribbon-constrained ~/templates/HCP_S1200_Atlas_Z4_pkXDZ/S1200.R.white_MSMAll.32k_fs_LR.surf.gii  \
#     ~/templates/HCP_S1200_Atlas_Z4_pkXDZ/S1200.R.pial_MSMAll.32k_fs_LR.surf.gii -greedy

# fslmaths $PATH_SOURCE/volume.1.L.nii -add $PATH_SOURCE/volume.1.R.nii $PATH_SOURCE/principal_gradient_MNI.nii.gz
