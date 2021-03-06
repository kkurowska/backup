import numpy as np
import matplotlib.pyplot as plt

import pywt.data
from PIL import Image
from scipy import ndimage
import scipy.misc

wavelet_name = 'haar'

# Load image
original = pywt.data.camera()

# im = np.zeros((256, 256)) # numpy square
# im[64:-64, 64:-64] = 1
# im = ndimage.rotate(im, 15, mode='constant') # diamond
# # im = ndimage.gaussian_filter(im, sigma=s)
# original = im

# original_image = Image.open('medical/' + image_name + '.jpg').convert('L')
# # original_image = ndimage.gaussian_filter(original_image, sigma=s)
# original = np.asarray(original_image, dtype="int32")

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation (LL)', ' Horizontal detail (LH)',
          'Vertical detail (HL)', 'Diagonal detail (HH)']

coeffs2 = pywt.dwt2(original, wavelet_name)
LL, (LH, HL, HH) = coeffs2

thresholds = [0, 0, 0, 0]
level = 95
fig = plt.figure()
for i, a in enumerate([LL, LH, HL, HH]):
    if i == 0:
        thresholds[i] = np.percentile(a, 100) + 0.01
    elif i == 3:
        thresholds[i] = np.percentile(a, level)
    else:
        thresholds[i] = np.percentile(a, level)
    a = np.flip(a, 0)
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(a, origin='image', interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)
    ax.set_axis_off()

coefs = [0, 0, 0, 0]
fig = plt.figure()
for i, a in enumerate([LL, LH, HL, HH]):
    threshold_mode = 'soft'
    da = pywt.threshold(a, thresholds[i], mode=threshold_mode)
    coefs[i] = da
    da = np.flip(da, 0)
    ax = fig.add_subplot(2, 2, i + 1)
    if i == 0:
        ax.imshow(da, origin='image', interpolation="nearest", cmap=plt.cm.gray, vmin=-1, vmax=1)
    else:
        ax.imshow(da, origin='image', interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)
    ax.set_axis_off()

denoised_coeffs2 = coefs[0], (coefs[1], coefs[2], coefs[3])

# Now reconstruct and plot the original image
reconstructed = pywt.idwt2(denoised_coeffs2, wavelet_name)

r2 = np.abs(reconstructed)

# scipy.misc.toimage(original).save('medical/results/' + image_name + '.png')
# scipy.misc.toimage(reconstructed).save('graphs/' + image_name + '_' + wavelet_name + '_' + str(level) + '.png')
# scipy.misc.toimage(r2).save('medical/results/' + image_name + '_' + wavelet_name + '_100_' + str(level) + '.png')

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_axis_off()
ax.set_title("original", fontsize=12)
plt.imshow(original, interpolation="nearest", cmap=plt.cm.gray)
ax = fig.add_subplot(1, 2, 2)
ax.set_axis_off()
ax.set_title("edge detected", fontsize=12)
plt.imshow(r2, interpolation="nearest", cmap=plt.cm.gray)

plt.show()
