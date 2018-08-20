import numpy as np
import matplotlib.pyplot as plt

import pywt.data
from PIL import Image
from scipy import ndimage
import scipy.misc

wavelet_name = 'haar'

s = 4
image_name = 'diamond_s' + str(s)

is_norm=False
b=50

# Load image
original = pywt.data.camera()

# im = np.zeros((256, 256))  # numpy square
# im[64:-64, 64:-64] = 1
# im = ndimage.rotate(im, 45, mode='constant')  # diamond
# im = ndimage.gaussian_filter(im, sigma=s)
# original = im

# original_image = Image.open(image_name + '.jpg').convert('L')
# original_image = ndimage.gaussian_filter(original_image, sigma=s)
# original = np.asarray(original_image, dtype="int32")

# fig = plt.figure()
# plt.hist(original, bins=b, normed=is_norm)

coeffs2 = pywt.dwt2(original, wavelet_name)
LL, (LH, HL, HH) = coeffs2

t = [0, 0, 0, 0]
l = 95
for i, a in enumerate([LL, LH, HL, HH]):
    if i == 0:
        t[i] = np.percentile(a, 100) + 0.01
    elif i == 3:
        t[i] = np.percentile(a, l)
    else:
        t[i] = np.percentile(a, l)

    if i != 0:
        fig = plt.figure()
        plt.hist(a, bins=b, normed=is_norm)

print('quantiles', t)

if False:
    coefs = [0, 0, 0, 0]
    for i, a in enumerate([LL, LH, HL, HH]):
        th_mode = 'soft'
        da = pywt.threshold(a, t[i], mode=th_mode)
        coefs[i] = da
        if i != 0:
            fig = plt.figure()
            plt.hist(da, bins=b, normed=is_norm)

    denoised_coeffs2 = coefs[0], (coefs[1], coefs[2], coefs[3])
    reconstructed = pywt.idwt2(denoised_coeffs2, wavelet_name)

    fig = plt.figure()
    plt.hist(reconstructed, bins=b, normed=is_norm)

plt.show()