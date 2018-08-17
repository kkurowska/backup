import numpy as np
import matplotlib.pyplot as plt

import pywt.data
from PIL import Image
from scipy import ndimage
import scipy.misc

wavelet_name = 'db1'

s = 8
image_name = 'koniczyna'

# Load image
# original = pywt.data.camera()

# im = np.zeros((256, 256)) # numpy square
# im[64:-64, 64:-64] = 1
# im = ndimage.rotate(im, 45, mode='constant') # diamond
# im = ndimage.gaussian_filter(im, sigma=s)
# original = im

original_image = Image.open(image_name + '.jpg').convert('L')
# original_image = ndimage.gaussian_filter(original_image, sigma=s)
original = np.asarray(original_image, dtype="int32")

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation (LL)', ' Horizontal detail (LH)',
          'Vertical detail (HL)', 'Diagonal detail (HH)']

coeffs2 = pywt.dwt2(original, wavelet_name)
LL, (LH, HL, HH) = coeffs2
t = [0, 0, 0, 0]
l = 95
fig = plt.figure()
for i, a in enumerate([LL, LH, HL, HH]):
    if i == 0:
        t[i] = np.percentile(a, 100) + 0.01
    elif i == 3:
        t[i] = np.percentile(a, l)
    else:
        t[i] = np.percentile(a, l)
    a = np.flip(a, 0)
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(a, origin='image', interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)
    ax.set_axis_off()
# fig.suptitle("2-D DWT Coefficients", fontsize=14)

# hist, bin_edges = np.histogram(LH, bins=60)
# plt.plot(bin_edges, hist)
# plt.show()

coefs = [0, 0, 0, 0]
fig = plt.figure()
for i, a in enumerate([LL, LH, HL, HH]):
    th_mode = 'soft'
    da = pywt.threshold(a, t[i], mode=th_mode)
    coefs[i] = da
    da = np.flip(da, 0)
    ax = fig.add_subplot(2, 2, i + 1)
    if i==0:
        ax.imshow(da, origin='image', interpolation="nearest", cmap=plt.cm.gray, vmin=-1, vmax=1)
    else:
        ax.imshow(da, origin='image', interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)
    ax.set_axis_off()
# fig.suptitle("Denoised Coefficients", fontsize=14)

denoised_coeffs2 = coefs[0], (coefs[1], coefs[2], coefs[3])

# Now reconstruct and plot the original image
reconstructed = pywt.idwt2(denoised_coeffs2, wavelet_name)
# print(np.min(reconstructed))
# print(np.max(reconstructed))
# change contrast
r1 = 255 - (np.sqrt(reconstructed / 255) * 255)

r2 = (np.abs(reconstructed) - 128) * 2 # from article

# print(np.min(r2))
# print(np.max(r2))

# image = Image.fromarray(r1).convert('L')
# image.save(image_name + '_' + wavelet_name + '.png')

# scipy.misc.toimage(original).save('graphs/' + image_name + '.png')
# scipy.misc.toimage(reconstructed).save('graphs/' + image_name + '_' + wavelet_name + '_' + str(l) + '.png')
# scipy.misc.toimage(r2).save('graphs/' + image_name + '_' + wavelet_name + '_' + str(l) + '_pp.png')

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
ax.set_axis_off()
plt.imshow(original, interpolation="nearest", cmap=plt.cm.gray)
ax = fig.add_subplot(2, 2, 2)
ax.set_axis_off()
plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)
ax = fig.add_subplot(2, 2, 3)
ax.set_axis_off()
plt.imshow(r1, interpolation="nearest", cmap=plt.cm.gray)
ax = fig.add_subplot(2, 2, 4)
ax.set_axis_off()
plt.imshow(r2, interpolation="nearest", cmap=plt.cm.gray)

plt.show()
