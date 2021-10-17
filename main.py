import cv2
import numpy as np
import os
path = 'imgs'
files = os.listdir(path)

for i in files:
    print(i)
    img = cv2.imread(f'imgs/{i}', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('.', img)
    n_white_pix = np.sum(img >= 165)
    n_pix = np.sum(img <= 255)
    cloud_percent = (n_white_pix/n_pix) * 100
    print(f'Number of white pixels in {i}:', n_white_pix)
    print(f'Number of pixels in {i}:', n_pix)
    print(f'Cloud percentage in {i}:', (cloud_percent), '%')