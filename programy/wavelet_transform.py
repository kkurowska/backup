import numpy as np
import matplotlib.pyplot as plt

import pywt.data
from PIL import Image
from scipy import ndimage
import scipy.misc

wavelet_name = 'db2'

image_name = 'white_square_15'

# Load image
# original = pywt.data.camera()

im = np.zeros((256, 256)) # square
im[64:-64, 64:-64] = 1
im = ndimage.rotate(im, 15, mode='constant')
original = im

# original_image = Image.open(image_name + '.jpg').convert('L')
# original = np.asarray(original_image, dtype="int32")

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']

coeffs2 = pywt.dwt2(original, wavelet_name)
LL, (LH, HL, HH) = coeffs2
t = [0, 0, 0, 0]
fig = plt.figure()
for i, a in enumerate([LL, LH, HL, HH]):
    if i == 0:
        t[i] = np.percentile(a, 100)
    elif i == 3:
        t[i] = np.percentile(a, 95)
    else:
        t[i] = np.percentile(a, 90)
    a = np.flip(a, 0)
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(a, origin='image', interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)
    ax.set_axis_off()
fig.suptitle("dwt2 coefficients", fontsize=14)


coefs = [0, 0, 0, 0]
fig = plt.figure()
for i, a in enumerate([LL, LH, HL, HH]):
    da = pywt.threshold(a, t[i], mode='soft')
    coefs[i] = da
    da = np.flip(da, 0)
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(da, origin='image', interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)
    ax.set_axis_off()
fig.suptitle("denoised coefficients", fontsize=14)

denoised_coeffs2 = coefs[0], (coefs[1], coefs[2], coefs[3])

# Now reconstruct and plot the original image
reconstructed = pywt.idwt2(denoised_coeffs2, wavelet_name)
print(np.min(reconstructed))
print(np.max(reconstructed))
# change contrast
r1 = 255 - (np.sqrt(reconstructed / 255) * 255)

r2 = (np.abs(reconstructed) - 128) * 2 # from article

print(np.min(r2))
print(np.max(r2))

# image = Image.fromarray(r2).convert('L')
# image.save(image_name + '_' + wavelet_name + '.png')

scipy.misc.toimage(r2).save(image_name + '_' + wavelet_name + '.png')

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
