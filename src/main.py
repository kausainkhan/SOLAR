import cv2
import numpy as np
import os
path = 'imgs'
files = os.listdir(path)

# for i in files:
#     print(i)
#     a = a + 1
#     for j in range(300):
#         print(j)

img = cv2.imread(f'imgs/130.jpg', cv2.IMREAD_GRAYSCALE)
ret, thresh1 = cv2.threshold(img, 87, 255, cv2.THRESH_BINARY)
cv2.imshow('.', thresh1)
cv2.waitKey()
n_white_pix = np.sum(thresh1 >= 245)
n_pix = np.sum(np.logical_and(25 <= img, img <= 255))
cloud_percent = (n_white_pix/n_pix) * 100
print(f'Number of white pixels in :', n_white_pix)
print(f'Number of pixels in :', n_pix)
print(f'Cloud percentage in :', cloud_percent, '%')