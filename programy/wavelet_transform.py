import numpy as np
import matplotlib.pyplot as plt

import pywt.data
from PIL import Image

wavelet_name = 'db3'

# Load image
# original = pywt.data.camera()

original_image = Image.open("black-square2.png").convert('L')
original = np.asarray(original_image, dtype="int32")

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
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(a, origin='image', interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)
fig.suptitle("dwt2 coefficients", fontsize=14)

# plt.figure()
# plt.hist(LL, bins=50)

coefs = [0, 0, 0, 0]
fig = plt.figure()
for i, a in enumerate([LL, LH, HL, HH]):
    da = pywt.threshold(a, t[i], mode='soft')
    # da = 255 - np.sqrt(da/255) * 255
    coefs[i] = da
    # print(np.max(da))
    # print(np.min(da))
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(da, origin='image', interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)
fig.suptitle("denoised coefficients", fontsize=14)

denoised_coeffs2 = coefs[0], (coefs[1], coefs[2], coefs[3])

# Now reconstruct and plot the original image
reconstructed = pywt.idwt2(denoised_coeffs2, wavelet_name)
reconstructed = 255 - (np.sqrt(reconstructed / 255) * 255)
fig = plt.figure()
plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)

# fig = plt.figure()
# plt.imshow(original, interpolation="nearest", cmap=plt.cm.gray)

plt.show()
