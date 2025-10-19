from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import numpy as np

def dice_score(pred, gt):
    return 2*np.sum(pred*gt)/(np.sum(pred)+np.sum(gt)+1e-8)

def jaccard_score(pred, gt):
    inter = np.sum(pred*gt)
    return inter/(np.sum(pred)+np.sum(gt)-inter+1e-8)

def evaluate(pred, gt):
    psnr = peak_signal_noise_ratio(gt, pred)
    ssim = structural_similarity(gt, pred)
    dice = dice_score(pred>0.5, gt>0.5)
    jacc = jaccard_score(pred>0.5, gt>0.5)
    print(f"PSNR={psnr:.2f}, SSIM={ssim:.2f}, Dice={dice:.3f}, Jaccard={jacc:.3f}")
