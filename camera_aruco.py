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
        0: [0.3, 0.29, 0.0],
        1: [-0.45, 0.29, 0.0],
        2: [-0.3, -1.13, 0.0],
        3: [0.29, -1.13, 0.0],
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
    marker_count = len(refMarkerArray)
    if marker_count < 4:  # 标志板少于4个
        raise RuntimeError('at least 3 pair of points required when invoking solvePnP')

    corners = corners
    ids = markerIDs
    # print('ids:\n')
    # print(ids)
    # print('corners:\n')
    # print(corners)

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

    # 如果檢測到的基準參考點大於3個，可以解算相機的姿態啦
    if len(objectPoints) >= 3:
        # 至少需要3個點
        retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMtx, dist)
        rMatrix, jacobian = cv2.Rodrigues(rvec)
        return True, rMatrix, tvec
    else:
        return False, None, None

    # 返回值
    # return rMatrix=[], tVecs=[]


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
    if len(targetsWorldPoint) < 2:
        print("检测到车辆少于2")
        return 0, 0, None
    if abs(targetsWorldPoint[0][0]) < 8 and abs(targetsWorldPoint[0][1]) < 8 and abs(targetsWorldPoint[1][0]) < 8 and \
            abs(targetsWorldPoint[1][1]) < 8:
        if targetsWorldPoint[0][0] > 0.5 or targetsWorldPoint[0][1] > 0.5 or targetsWorldPoint[0][1] > 0.5 \
                or targetsWorldPoint[0][1] > 0.5:
            print("已过十字路口")
            return 0, 0, None
        else:
            D1, D2, T11, T12, T21, T22 = calculateTime(targetsWorldPoint, target1Point, target2Point)
            if T11 < 0 or T12 < 0 or T21 < 0 or T22 < 0:
                return 0, 0, None
            if abs(T11) < 10 and abs(T21) < 10:
                if abs(T12) < abs(T21) or abs(T11) > abs(T22):
                    return 0, 0, None
                else:
                    if T11 < T22 < T12:
                        # 1车减速，使2车通过
                        spe = changeSpeed(D1, T22, carLength=0.5, carWidth=0.26, index=0.5, weightIndex=1.2)
                        return 1, 1, spe
                    if T21 < T12 < T22:
                        # 2车减速，使1车通过
                        spe = changeSpeed(D2, T12, carLength=0.5, carWidth=0.26, index=0.2, weightIndex=1.2)
                        return 1, 2, spe
            else:
                return 0, 0, None
    else:
        print("已超出预警范围")
        return 0, 0, None


def calculateTime(targetsWorldPoint, target1Point, target2Point):
    target1Point.append(targetsWorldPoint[0])
    target2Point.append(targetsWorldPoint[1])
    D1, V1 = calculateDistanceAndV(target1Point, frameN=3)
    D2, V2 = calculateDistanceAndV(target2Point, frameN=3)
    T11, T12 = calculateT(D1, V1, carLength=0.5, carWidth=0.26, index=0.5)
    T21, T22 = calculateT(D2, V2, carLength=0.5, carWidth=0.26, index=0.2)
    # print(T11, T12, T21, T22)
    return D1, D2, T11, T12, T21, T22


def calculateDistanceAndV(targetPoint, frameN):
    if len(targetPoint) >= frameN + 1:
        Vx = abs(targetPoint[len(targetPoint) - 1][0] - targetPoint[len(targetPoint) - (frameN + 1)][0]) * 20 / frameN
        Vy = abs(targetPoint[len(targetPoint) - 1][1] - targetPoint[len(targetPoint) - (frameN + 1)][1]) * 20 / frameN
        if Vx > Vy:
            distance = abs(targetPoint[len(targetPoint) - 1][0])
            V = Vx
        else:
            distance = abs(targetPoint[len(targetPoint) - 1][1])
            V = Vy
        return distance, V
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
    return SPeed


def cameraAruco():
    target1Point, target2Point = [], []

    global warning, AIM, SPE
    mtx, dist, rMatrix, tvec, refMarkerArray, targetMarker = parameterPrepare()

    vc = cv2.VideoCapture("./vehicle726.MP4")
    # vc = cv2.VideoCapture(0)
    while True:
        # time_start = time.time()

        ret1, frame = vc.read()
        if ret1 == 0:
            break

        # 1. 估計camera pose
        # 1.1 detect aruco markers
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters)
        # aruco.drawDetectedMarkers(img_gray, corners, ids)  # Draw A square around the markers

        img = frame.copy()
        aruco.drawDetectedMarkers(img, corners, ids)
        cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('detect', 450, 800)
        cv2.imshow("detect", img)
        cv2.waitKey(0)

        # 1.2 estimate camera pose
        gotCameraPose, rMatrixTemp, tvecTemp = estimateCameraPose(mtx, dist, refMarkerArray, corners, ids)

        # 1.3 update R, T to static value
        if gotCameraPose:
            rMatrix = rMatrixTemp
            tvec = tvecTemp
            # print('rMatrix\n' + str(rMatrixTemp))
            # print('tvec\n' + str(tvecTemp))

        # 2. 根據目標的marker來計算世界坐標系坐標
        targetsWorldPoint = detectTarget(mtx, dist, rMatrix, tvec, targetMarker, corners, ids)
        # print(targetsWorldPoint)

        # 3. 根據目標世界坐標判断是否有相撞风险
        warning, AIM, SPE = judgeWarning(targetsWorldPoint, target1Point, target2Point)
        if warning == 1:
            print("WarningInformation: " + str(warning) + "    第 " + str(AIM) + " 辆车需调速为 " + str(SPE)[0: 4])
        if warning == 0:
            print("WarningInformation: " + str(warning))
        # print('------------------------------')
        # print(time.time() - time_start)
        '''
        if ( cv2.waitKey(10) & 0xFF ) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
        '''
        # cap.release()
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    warning, AIM, SPE = 0, 0, None
    cameraAruco()
