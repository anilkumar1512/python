import cv2 as cv
import numpy as np

img = cv.imread(r"C:\Users\perur\OneDrive\Desktop\Fig0457(a)(thumb_print).tif", 0)
rows, columns = img.shape
a = 50
x = 4

dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

u = np.arange(rows)
v = np.arange(columns)
u, v = np.meshgrid(u - rows//2, v - columns//2, indexing='ij')
d = np.sqrt(u**2 + v**2)

h = 1 / (1 + (a / (d + 1e-5))**(2 * x))
filtered = dft_shift * h

f_ishift = np.fft.ifftshift(filtered)
img_back = np.fft.ifft2(f_ishift)
img_back1 = np.abs(img_back)
img_back1 = np.uint8(np.clip(img_back1, 0, 255))

cv.imshow('highpass filtered ', img_back1)
cv.waitKey(0)

thresholded = np.where(img_back > 0, 255, 0).astype(np.uint8)
cv.imshow('thresholded image', thresholded)
cv.waitKey(0)

