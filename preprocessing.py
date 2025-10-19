import nibabel as nib
import numpy as np

def normalize_intensity(volume):
    v = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
    v = np.clip(v, -3, 3)
    return (v - v.min()) / (v.max() - v.min())

def skull_strip(volume, threshold=0.1):
    mask = volume > threshold
    volume[~mask] = 0
    return volume

def load_nifti(path):
    return nib.load(path).get_fdata()

def save_nifti(data, path):
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
