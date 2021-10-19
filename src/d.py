import cv2
import numpy as np

assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3, 'OpenCV version 3 or newer required.'

src    = cv2.imread("imgs/0.jpg")
width  = src.shape[1]
height = src.shape[0]

f =  width /(2 * np.tan(np.array(160 * 3.14 / 360)))
Cu = width/2
Cv = height/2

K = np.array([[f, 0, Cu],
     [0, f, Cv],
     [0, 0, 1 ]])

# zero distortion coeffinciets work well for this image
D = np.array([0., 0., 0., 0.])

# # use Knew to scale the output
# Knew = K.copy()
# Knew[(0,1), (0,1)] = 0.4 * Knew[(0,1), (0,1)]

print(width, height, Cu, Cv, D, f)
img = cv2.imread('imgs/0.jpg')
img_undistorted = cv2.fisheye.undistortImage(img, K, D=D)
cv2.imwrite('fisheye_sample_undistorted.jpg', img_undistorted)
cv2.imshow('undistorted', img_undistorted)
cv2.waitKey()

# import numpy as np
# import cv2

# src    = cv2.imread("imgs/0.jpg")
# width  = src.shape[1]
# height = src.shape[0]

# distCoeff = np.zeros((4,1),np.float64)

#   # TODO: add your coefficients here!
# k1 = -1.0e-5; # negative to remove barrel distortion
# k2 = 0.0;
# p1 = 0.0;
# p2 = 0.0;

# distCoeff[0,0] = k1;
# distCoeff[1,0] = k2;
# distCoeff[2,0] = p1;
# distCoeff[3,0] = p2;

#   # assume unit matrix for camera
# cam = np.eye(3,dtype=np.float32)

# cam[0,2] = width/2.0  # define center x
# cam[1,2] = height/2.0 # define center y
# cam[0,0] = 10.        # define focal length x
# cam[1,1] = 10.        # define focal length y

#   # here the undistortion will be computed
# dst = cv2.undistort(src,cam,distCoeff)

# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()