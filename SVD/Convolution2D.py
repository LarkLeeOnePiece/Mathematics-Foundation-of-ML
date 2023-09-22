import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 1, 0, 1, 1, 0, 1, 1],
             [1, 1, 1, 0, 1, 1, 0, 1, 1],
             [1, 1, 1, 0, 1, 1, 0, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1]])


plt.imshow(X, cmap='gray')

# 1.66 scipy.signal 实现图像的二维卷积
img = X
kernel1 = np.array([[0.25,0.25],[0.25,0.25]])
kernel2 = np.array([[-1,1],[0,0]])
kernel3 = np.array([[-1,0],[1,0]])
kernel4 = np.array([[-1,2,-1],[-1,2,-1]])  #
kernel5 = np.array([[-1,-1],[2,2],[-1,-1]])
# scipy.signal 实现卷积运算
from scipy import signal

convSame1 = signal.convolve2d(img, kernel1, boundary='symm', mode='same')  # same 卷积
convSame2 = signal.convolve2d(img, kernel2, boundary='symm', mode='same')  # same 卷积
convSame3 = signal.convolve2d(img, kernel3, boundary='symm', mode='same')  # same 卷积
convSame4 = signal.convolve2d(img, kernel4, boundary='symm', mode='same')  # same 卷积
convSame5 = signal.convolve2d(img, kernel5, boundary='symm', mode='same')  # same 卷积
plt.figure()
plt.subplot(231), plt.title('Original')
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.subplot(232), plt.title('kernel-1')
plt.imshow(np.absolute(convSame1), cmap='gray')
plt.subplot(233), plt.title('kernel-2')
plt.imshow(np.absolute(convSame2), cmap='gray')
plt.subplot(234), plt.title('kernel-3')
plt.imshow(np.absolute(convSame3), cmap='gray')
plt.subplot(235), plt.title('kernel-4')
plt.imshow(np.absolute(convSame4), cmap='gray')
plt.subplot(236), plt.title('kernel-5')
plt.imshow(np.absolute(convSame5), cmap='gray')
plt.tight_layout()
plt.show()

# YOUR CODE GOES HERE
