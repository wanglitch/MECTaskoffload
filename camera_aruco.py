#!/usr/bin/ python
# -*- coding: utf-8 -*-

# 使用視覺方法測量目標在世界坐標系中的坐標
# 首先估計相機姿態,然後測算目標marker中心點在世界坐標系中的位置.
# 使用方法:
# 1. 相機校準,
# 2. 在空間中放置4個以上的基準坐標點,在程序中給定這些點的信息,包括ID和世界坐標
# 3. 被測目標使用marker標記,在程序中給定這些點的markerID
# 4. 拍攝錄像,確保4個標志點在視野內.
# 5. 運行程序處理視頻幀
# CR@ Guofeng, mailto:gf@gfshen.cn
#
# ------版本歷史---
# ---V1.0
# ---2019年7月19日
#    初次編寫
import socket
import threading
import time
import numpy as np
import cv2
import cv2.aruco as aruco


# 实时获取最新帧，解决由于图像处理速度不够等原因造成帧堆积的问题
class ThreadedCamera(object):
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.status, self.frame = False, None

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def grab_frame(self):
        if self.status:
            return self.frame
        return None


def parameterPrepare():
    try:
        npzfile = np.load('./calibrateData920.npz')
        mtx = npzfile['mtx']
        dist = npzfile['dist']
    except IOError:
        raise Exception('cant find calibration data, do that first')

    # 保存基準點的信息,檢測到之後會更新.
    rMatrix, tvec = [], []

    # 0.1. 指定基準點的marker ID和世界坐標
    # [[marker ID, X, Y, Z]..]
    # refMarkerArray = {
    #     0: [0.385, -0.45, 0.0],
    #     1: [0.385, 0.45, 0.0],
    #     2: [-0.69, -1.05, 0.0],
    #     3: [-0.69, 0.45, 0.0],
    # }

    refMarkerArray = {
        0: [-0.72, -0.7, 0.05],
        1: [0.72, -0.7, 0.0],
        2: [-0.72, 0.68, 0.05],
        3: [0.72, 0.68, 0.0],
    }

    # 0.2 指定目標的marker ID
    targetMarker = [10, 11]
    return mtx, dist, rMatrix, tvec, refMarkerArray, targetMarker


def estimateCameraPose(cameraMtx, dist, refMarkerArray, corners, markerIDs):
    """
    根据基准点的marker，解算相机的旋转向量rvecs和平移向量tvecs，(solvePnP(）实现)
    并将rvecs转换为旋转矩阵输出(通过Rodrigues())
    输入：
        cameraMtx内参矩阵，
        dist畸变系数。
        当前处理的图像帧frame，
        用于定位世界坐标系的参考点refMarkerArray.  py字典类型,需要len(refMarkerArray)>=3, 格式：{ID:[X, Y, Z], ID:[X,Y,Z]..}
        corners, detectMarkers()函數的輸出
        markerIDs, detectMarkers()函數的輸出
    输出：旋转矩阵rMatrix, 平移向量tVecs
    """
    ids = markerIDs
    objectPoints, imagePoints = [], []
    # 检查是否探测到了所有预期的基准marker
    if len(ids) != 0:  # 檢測到了marker,存儲marker的世界坐標到objectPoints，構建對應的圖像平面坐標列表 imagePoints
        # print('------detected ref markers----')
        for i in range(len(ids)):  # 遍歷探測到的marker ID,
            if ids[i][0] in refMarkerArray:  # 如果是參考點的標志，提取基准点的图像坐标，用于构建solvePnP()的输入
                # print('id:\n ' + str(ids[i][0]))
                # print('cornors: \n ' + str(corners[i][0]))
                objectPoints.append(refMarkerArray[ids[i][0]])
                imagePoints.append(corners[i][0][0].tolist())  # 提取marker的左上點
        objectPoints = np.array(objectPoints)
        imagePoints = np.array(imagePoints)
        # print('------------------------------\n')
        # print('objectPoints:\n' + str(objectPoints))
        # print('imagePoints:\n' + str(imagePoints))
        pass
    else:
        return False, None, None

    # 如果檢測到的基準參考點大於4個，可以解算相機的姿態
    if len(objectPoints) >= 4:
        # 至少需要3個點
        retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMtx, dist)
        rMatrix, jacobian = cv2.Rodrigues(rvec)
        return True, rMatrix, tvec
    else:
        return False, None, None


def detectTarget(cameraMatrix, dist, rMatrix, tvec, targetMarker, corners, markerIDs):
    """
    測算目標marker中心在世界坐標系中的位置
    輸入:
    輸出:
        與markerIDs長度相等的列表,包含位置確定的目標坐標,未檢測到填None,例如[None,[x2,y2,z2]]
    """
    if not rMatrix.any():
        return
    targets_count = len(targetMarker)
    if targets_count == 0:
        raise Exception('targets empty, are you dou?')

    # 創建與targetMarker相同尺寸的列表,用於存儲解算所得到目標的世界坐標
    targetsWorldPoint = [None] * targets_count

    for i in range(len(markerIDs)):  # 遍歷探測到的marker ID,
        markerIDThisIterate = markerIDs[i][0]
        if markerIDThisIterate in targetMarker:  # 如果是目標marker的ID
            # 獲得當前處理的marker在targetMarker中的下標,用於填充targetsWorldPoint
            targetIndex = targetMarker.index(markerIDThisIterate)
            # print(targetIndex)
        else:
            continue

        # 計算marker中心的圖像坐標
        markerCenter = corners[i][0].sum(0) / 4.0
        # 畸變較正,轉換到相機坐標系,得到(u,v,1)
        # https://stackoverflow.com/questions/39394785/opencv-get-3d-coordinates-from-2d
        markerCenterIdeal = cv2.undistortPoints(markerCenter.reshape([1, -1, 2]), cameraMatrix, dist)
        markerCameraCoodinate = np.append(markerCenterIdeal[0][0], [1])
        # print('++++++++markerCameraCoodinate')
        # print(markerCameraCoodinate)

        # marker的坐標從相機轉換到世界坐標
        markerWorldCoodinate = np.linalg.inv(rMatrix).dot((markerCameraCoodinate - tvec.reshape(3)))
        # print('++++++++markerworldCoodinate')
        # print(markerWorldCoodinate)
        # 將相機的坐標原點轉換到世界坐標系
        originWorldCoodinate = np.linalg.inv(rMatrix).dot((np.array([0, 0, 0.0]) - tvec.reshape(3)))
        # 兩點確定了一條直線 (x-x0)/(x0-x1) = (y-y0)/(y0-y1) = (z-z0)/(z0-z1)
        # 當z=0時,算得x,y
    # 返回值
    # return rMatrix=[], tVecs=[]
        delta = originWorldCoodinate - markerWorldCoodinate

        # 给定标识高度
        if targetIndex == 0:
            zWorld = 0.17
        elif targetIndex == 1:
            zWorld = 0.30
        else:
            zWorld = 0.0

        xWorld = (zWorld - originWorldCoodinate[2]) / delta[2] * delta[0] + originWorldCoodinate[0]
        yWorld = (zWorld - originWorldCoodinate[2]) / delta[2] * delta[1] + originWorldCoodinate[1]
        targetsWorldPoint[targetIndex] = [xWorld, yWorld, zWorld]

        # print('-=-=-=\n Target Position ' + str(targetsWorldPoint[targetIndex]))
        print(' Target ' + str(targetIndex + 1) + ' Position ' + str(targetsWorldPoint[targetIndex]))

    # print(time.time() - time_start)
    return targetsWorldPoint


def judgeWarning(targetsWorldPoint, target1Point, target2Point):
    if targetsWorldPoint[0] is None or targetsWorldPoint[1] is None:
        print("检测到车辆少于二，停止预警")
        return "00-9.9"
    elif abs(targetsWorldPoint[0][0]) < 8 and abs(targetsWorldPoint[0][1]) < 8 and \
            abs(targetsWorldPoint[1][0]) < 8 and abs(targetsWorldPoint[1][1]) < 8:
        if targetsWorldPoint[0][0] > 0.6 or targetsWorldPoint[0][1] > 0.6 or targetsWorldPoint[0][1] > 0.6 \
                or targetsWorldPoint[0][1] > 0.6:
            print("已过十字路口，停止预警")
            return "00-9.9"
        else:
            D1, D2, T11, T12, T21, T22 = calculateTime(targetsWorldPoint, target1Point, target2Point)
            if 0 < T11 < 10 and 0 < T21 < 10:
                if T11 < T22 < T12:
                    # 1车减速，使2车通过
                    spe = changeSpeed(D1, T22, carLength=0.5, carWidth=0.32, index=0.5, weightIndex=1.2)
                    return "11" + spe
                elif T21 < T12 < T22:
                    # 2车减速，使1车通过
                    spe = changeSpeed(D2, T12, carLength=0.5, carWidth=0.32, index=0.2, weightIndex=1.2)
                    return "12" + spe
                else:
                    return "00-9.9"
            else:
                return "00-9.9"
    else:
        print("已超出预警范围")
        return "00-9.9"


def calculateTime(targetsWorldPoint, target1Point, target2Point):
    target1Point.append(targetsWorldPoint[0])
    target2Point.append(targetsWorldPoint[1])
    D1, V1 = calculateDistanceAndV(target1Point, frameN=3, flag="x")
    D2, V2 = calculateDistanceAndV(target2Point, frameN=3, flag="y")
    T11, T12 = calculateT(D1, V1, carLength=0.5, carWidth=0.32, index=0.5)
    T21, T22 = calculateT(D2, V2, carLength=0.5, carWidth=0.32, index=0.2)
    # print(D1, D2, T11, T12, T21, T22)
    return D1, D2, T11, T12, T21, T22


def calculateDistanceAndV(targetPoint, frameN, flag):
    if len(targetPoint) >= frameN + 1:
        if flag == "x":
            Vx = abs(targetPoint[len(targetPoint) - 1][0] - targetPoint[len(targetPoint) - (frameN+1)][0]) * 30 / frameN
            distance = abs(targetPoint[len(targetPoint) - 1][0])
            return distance, Vx
        elif flag == "y":
            Vy = abs(targetPoint[len(targetPoint) - 1][1] - targetPoint[len(targetPoint) - (frameN+1)][1]) * 30 / frameN
            distance = abs(targetPoint[len(targetPoint) - 1][1])
            return distance, Vy
    else:
        return 999.99, 0.0


def calculateT(distance, V, carLength, carWidth, index):
    if V == 0:
        return 999.99, 999.99
    T1 = (distance - carLength * index - carWidth / 2) / V
    T2 = (distance + carLength * (1 - index) + carWidth / 2) / V
    return T1, T2


def changeSpeed(distance, T, carLength, carWidth, index, weightIndex):
    SPeed = (distance - carLength * index - carWidth / 2) / T / weightIndex
    return str(SPeed)[0:4]
