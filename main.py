#! coding=utf-8
import ctypes
import socket
import threading
import time
import cv2
import cv2.aruco as aruco
import shm
import taskImageDetect
from camera_aruco import detectTarget, judgeWarning, estimateCameraPose, parameterPrepare
import multiprocessing

# 进程间通信所用端口：5214


def cameraAruco(warningInfo):
    target1Point, target2Point = [], []
    mtx, dist, rMatrix, tvec, refMarkerArray, targetMarker = parameterPrepare()

    # vc = cv2.VideoCapture("./vehicle4k.mp4")
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    vc.set(cv2.CAP_PROP_FPS, 30)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        time_start = time.time()

        ret1, frame = vc.read()
        if ret1 == 0:
            break

        # 1. 估計camera pose
        # 1.1 detect aruco markers
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters)

        img = frame.copy()
        aruco.drawDetectedMarkers(img, corners, ids)
        cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('detect', 1280, 720)
        cv2.imshow("detect", img)
        cv2.waitKey(1)

        # 1.2 estimate camera pose
        gotCameraPose, rMatrixTemp, tvecTemp = estimateCameraPose(mtx, dist, refMarkerArray, corners, ids)

        # 1.3 update R, T to static value
        if gotCameraPose:
            rMatrix = rMatrixTemp
            tvec = tvecTemp

        # 2. 根據目標的marker來計算世界坐標系坐標
        targetsWorldPoint = detectTarget(mtx, dist, rMatrix, tvec, targetMarker, corners, ids)

        # 3. 根據目標世界坐標判断是否有相撞风险
        warningInfo.value = judgeWarning(targetsWorldPoint, target1Point, target2Point)
        if warningInfo.value[0] == "0":
            print("WarningInformation: " + warningInfo.value[0])
        else:
            print("WarningInformation: " + warningInfo.value[0] + \
                  "    第 " + warningInfo.value[1] + " 辆车需调速为 " + warningInfo.value[2:])
        print("TimeDelay: " + str(time.time() - time_start))
        print('------------------------------')


def listenAndSend(listenPort, sendPort, warningInfo):
    imageDetect = taskImageDetect.TaskImageDetect()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', listenPort))
    data, _ = s.recvfrom(1024)
    s.sendto(data, ('127.0.0.1', sendPort))
    numOfShm, idArray = shm.getShmInfo(data)
    for i in range(numOfShm):
        # t = threading.Thread(target=compTaskWith, args=(idArray[i], imageDetect, warningInfo,))
        t = multiprocessing.Process(target=compTaskWith, args=(idArray[i], imageDetect, warningInfo,))
        t.start()


def compTaskWith(ID, imageDealClass, warningInfo):
    while True:
        taskTarget, _, _, taskData = shm.getTheTaskIfThereIsOne(ID)
        if taskTarget == 1:
            lineColor, calComplete = "0", 3
        elif taskTarget == 2:
            lineColor, calComplete = "2", 4
        else:
            lineColor, calComplete = "0", 0

        # img = imageDealClass.getImgFromBytes(taskData)
        # cv2.namedWindow(str(taskTarget), cv2.WINDOW_NORMAL)
        # cv2.imshow(str(taskTarget), img)
        # cv2.waitKey(1)

        # 计算任务
        # out = imageDealClass.getTheLightColor(taskData)  # getTheLightColor
        # out = imageDealClass.getTheHuman(taskData)  # getTheHuman
        out = "no human"
        # print(out)

        # [24:32]human  [32:33]lineColor  [33:34]warning  [34:35]warningCarTarget  [35:39]suggestSpeed
        calResult_str = "{}".format(out + lineColor + warningInfo.value)
        calResult_byte = calResult_str.encode('utf-8')
        # 组织计算结果
        return_data = calComplete.to_bytes(1, 'little') + len(calResult_byte).to_bytes(2, 'little') + calResult_byte
        shm.writeCalResult(ID, return_data)


def main():
    # warningInfo:  |warning:aim:speed| <=> |0:1:2~6|
    warningInfo = multiprocessing.Manager().Value(ctypes.c_char_p, '"00-9.9"')
    p = multiprocessing.Process(target=cameraAruco, args=(warningInfo,))
    q = multiprocessing.Process(target=listenAndSend, args=(5214, 5215, warningInfo,))
    p.start()
    q.start()
    p.join()
    q.join()


if __name__ == '__main__':
    main()
