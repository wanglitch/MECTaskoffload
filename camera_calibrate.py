#!/usr/bin/ python
# -*- coding: utf-8 -*-
# 從文件夾讀取棋盤圖片,校準相機,將參數保存到文件.
# 2019年7月17日
# guofeng， mailto:gf@gfshen.cn
# ---------------------------------------
import numpy as np
import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt


# ref: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

# 8,6
# def calibrate(cornerX=5, cornerY=8, squareSize=30.0, images=glob.glob('*.png')):
def calibrate(cornerX=5, cornerY=8, squareSize=30.0):
    # termination criteria
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    global gray
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cornerX * cornerY, 3), np.float32)

    # 如果squareSize设错，对mtx和dist的估计没有影响，但对rvecs和tvecs有影响
    objp[:, :2] = np.mgrid[0:cornerX, 0:cornerY].T.reshape(-1, 2) * squareSize
    # objp[:, :2] = np.mgrid[0:cornerX, 0:cornerY].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    vc = cv2.VideoCapture("./chess9202.webm")
    while True:
        ret1, img = vc.read()
        if ret1 == 0:
            break
        # for fname in images:
        # img = cv2.imread(img)
        # plt.imshow(img)
        # plt.show()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # plt.imshow(gray, cmap="gray")
        # plt.show()

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cornerX, cornerY), None)
        print(ret)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (cornerX, cornerY), corners2, ret)
            cv2.namedWindow("img", 0)
            cv2.resizeWindow("img", 612, 816)
            cv2.imshow('img', img)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    # 开始标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print('mtx: \n', mtx)  # 内参数矩阵
        print('dist: \n', dist)  # 畸变系数
        # print('rvecs: \n', rvecs)
        # print('tvecs: \n',tvecs)
        np.savez('calibrateData920.npz', mtx=mtx, dist=dist)

    return ret, mtx, dist, rvecs, tvecs


def undistortion(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    if roi != (0, 0, 0, 0):
        dst = dst[y:y + h, x:x + w]

    return dst


if __name__ == '__main__':
    # image = glob.glob('D:/pythonProject/condabase/mychess/*.png')
    # if not image:
    #     raise RuntimeError('Cant find image existing')
    # for item in image:
    #     img = Image.open(item)
    #     plt.imshow(img)
    #     plt.show()

    # 用于存储内参矩阵和畸变参数
    mtx, dist = [], []
    try:
        npzfile = np.load('calibrate.npz')
        mtx = npzfile['mtx']
        dist = npzfile['dist']
    except IOError:
        # ret, mtx, dist, rvecs, tvecs = calibrate(cornerX=5, cornerY=8, squareSize=30.0, images=image)
        ret, mtx, dist, rvecs, tvecs = calibrate(cornerX=5, cornerY=8, squareSize=30.0)
    print('calibration finished, result saved as .npz file ')
