import numpy as np
from clustering import vpfcwbc
from preprocessing import save_nifti

X = np.load("features.npy")
labels, bias = vpfcwbc(X, n_clusters=4)
save_nifti(labels.reshape(256,256), "segmentation_result.nii")
