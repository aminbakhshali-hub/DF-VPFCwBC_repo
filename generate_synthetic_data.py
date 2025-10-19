import numpy as np, matplotlib.pyplot as plt
from preprocessing import save_nifti

def generate_image():
    img = np.zeros((256,256))
    img[40:180, 60:220] = 0.5
    img[80:140, 100:180] = 0.8
    bias = 0.2*np.sin(np.linspace(0, np.pi, 256)).reshape(256,1)
    img_biased = img + bias
    noise = np.random.normal(0, 0.05, img_biased.shape)
    img_noisy = np.clip(img_biased + noise, 0, 1)
    save_nifti(img_noisy, "synthetic.nii")
    plt.imshow(img_noisy, cmap='gray'); plt.title("Synthetic MR Image"); plt.show()

if __name__ == "__main__":
    generate_image()
