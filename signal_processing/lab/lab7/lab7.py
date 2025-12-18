from scipy import datasets, ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

curr_dir = os.path.dirname(__file__)
lab_dir = os.path.dirname(curr_dir)
plot_dir = os.path.join(curr_dir, "graphs")

img = datasets.face(gray=True)

def ex1():
    global img
    signal1 = np.array([np.array([np.sin(2 * np.pi * i + 3 * np.pi * j) for j in range(img.shape[1])]) for i in range(img.shape[0])])
    plt.imshow(signal1, cmap=plt.cm.gray)
    plt.savefig(f"{plot_dir}/ex1/a/signal.pdf")
    
    signal2 = np.array([np.array([np.sin(4 * np.pi * i) + np.cos(6 * np.pi * j) for j in range(img.shape[1])]) for i in range(img.shape[0])])
    plt.imshow(signal2, cmap=plt.cm.gray)
    plt.savefig(f"{plot_dir}/ex1/b/signal.pdf")

    freq_signal3 = np.zeros(img.shape)
    # Activate frequency 5 on the horizontal
    freq_signal3[0][5] = 1
    freq_signal3[0][-5] = 1

    signal3 = np.fft.ifft2(freq_signal3).real
    plt.imshow(signal3, cmap=plt.cm.gray)
    plt.savefig(f"{plot_dir}/ex1/c/signal.pdf")
    
    plt.imshow(20*np.log10(abs(freq_signal3 + 0.001)))
    plt.colorbar()
    plt.savefig(f"{plot_dir}/ex1/c/colorbar.pdf")

    freq_signal4 = np.zeros(img.shape)
    # Activate frequency 5 on vertical
    freq_signal4[5][0] = 1
    freq_signal4[-5][0] = 1

    signal4 = np.fft.ifft2(freq_signal4).real
    plt.imshow(signal4, cmap=plt.cm.gray)
    plt.savefig(f"{plot_dir}/ex1/d/signal.pdf")

    plt.imshow(20*np.log10(abs(freq_signal4 + 0.001)))
    plt.colorbar()
    plt.savefig(f"{plot_dir}/ex1/d/colorbar.pdf")

    freq_signal5 = np.zeros(img.shape)
    # Activate frequency 5 on vertical and horizontal
    freq_signal5[5][5] = 1
    freq_signal5[-5][-5] = 1

    signal5 = np.fft.ifft2(freq_signal5).real
    plt.imshow(signal5, cmap=plt.cm.gray)
    plt.savefig(f"{plot_dir}/ex1/e/signal.pdf")

    plt.imshow(20*np.log10(abs(freq_signal5 + 0.001)))
    plt.colorbar()
    plt.savefig(f"{plot_dir}/ex1/e/colorbar.pdf")

# Clip anything too far from the mean to the mean
def cut_based_on_mean(img):
    fft_img = np.fft.fft2(img)
    fft_mag = np.abs(fft_img)
    
    mag_mean = np.mean(fft_mag)
    mag_std = np.std(fft_mag)
    std_count = 3

    for i in range(len(fft_img)):
        for j in range(len(fft_img[i])):
            if fft_mag[i][j] - mag_mean > std_count * mag_std:
                fft_img[i][j] = mag_mean
    img_cutoff = np.fft.ifft2(fft_img)
    img_cutoff = np.real(img_cutoff)

    return img_cutoff

# Cuts upper half of the frequencies
def cut_high_freq_patterns(img):
    fft_img = np.fft.fft2(img)

    # '/' like patterns
    fft_img[fft_img.shape[0] // 4:fft_img.shape[0] // 2, fft_img.shape[1] // 4:fft_img.shape[1] // 2] = 0
    fft_img[-fft_img.shape[0] // 2:-fft_img.shape[0] // 4, -fft_img.shape[1] // 2:-fft_img.shape[1] // 4] = 0
    
    # Horziontal freq
    fft_img[:fft_img.shape[0] // 4, fft_img.shape[1] // 4:fft_img.shape[1] // 2] = 0
    fft_img[-fft_img.shape[0] // 4:, -fft_img.shape[1] // 2:-fft_img.shape[1] // 4] = 0
    
    # Vertical freq
    fft_img[fft_img.shape[0] // 4:fft_img.shape[0] // 2, :fft_img.shape[1] // 4] = 0
    fft_img[-fft_img.shape[0] // 2:-fft_img.shape[0] // 4, -fft_img.shape[1] // 4:] = 0

    # '\' pattern, pos row, neg col
    fft_img[fft_img.shape[0] // 4:fft_img.shape[0] // 2, -fft_img.shape[1] // 2:-fft_img.shape[1] // 4:] = 0
    fft_img[-fft_img.shape[0] // 2:-fft_img.shape[0] // 4, fft_img.shape[1] // 4:fft_img.shape[1] // 2:] = 0

    img_cut = np.fft.ifft2(fft_img)
    img_cut = np.real(img_cut)
                                    
    return img_cut

def ex2():
    global img
    img_cut = cut_based_on_mean(img)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.savefig(f"{plot_dir}/ex2/before.pdf")

    plt.imshow(img_cut, cmap=plt.cm.gray)
    plt.savefig(f"{plot_dir}/ex2/after.pdf")

def ex3():
    pixel_noise = 200
    noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=img.shape)
    img_noisy = img + noise
    noise_energy = np.linalg.norm(noise) ** 2
    
    plt.imshow(img_noisy, cmap=plt.cm.gray)
    plt.savefig(f"{plot_dir}/ex3/initial_img.pdf")
    print(f"Initial SNR: {np.linalg.norm(img_noisy) ** 2 / noise_energy}")

    cut_img = cut_high_freq_patterns(img_noisy)
    plt.imshow(cut_img, cmap=plt.cm.gray)
    plt.savefig(f"{plot_dir}/ex3/high_freq_cut.pdf")
    print(f"Cut big freq patterns SNR: {np.linalg.norm((cut_img)) ** 2 / noise_energy}")
    
    cut_img = cut_based_on_mean(img_noisy)
    plt.imshow(cut_img, cmap=plt.cm.gray)
    plt.savefig(f"{plot_dir}/ex3/mean_cut.pdf")
    print(f"Cut freq bigger than mean SNR: {np.linalg.norm(cut_img) ** 2 / noise_energy}")

ex1()
ex2()
ex3()
