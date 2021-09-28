#! coding=utf-8
import socket
import threading
import multiprocessing
import time
import cv2
import cv2.aruco as aruco
import shm
import taskImageDetect
# t = threading.Thread(target=loop, name='LoopThread')
# t.start()
from camera_aruco import detectTarget, judgeWarning, estimateCameraPose, parameterPrepare

deliMeter = "|:|:|"
warning, AIM = 0, 0
SPE = None


# 进程间通信所用端口：5214


# |1:
def getShmInfo(data):
    dataStr = data.decode("utf-8")
    data_split = dataStr.split(deliMeter)
    # print("0",data_split[0])
    # print("1",data_split[1])
    numOfShm = data[0]
    id_array = []
    print(data)
    for i in range(numOfShm):
        id_array.append(int(data_split[i + 1]))
    # print(id_array)
    return numOfShm, id_array


def compTaskWith(ID, imageDealClass):
    global warning, AIM, SPE
    while True:
        taskTarget, _, _, taskData = shm.getTheTaskIfThereIsOne(ID)
        if taskTarget == 1:
            lineColor = "0"
            calComplete = 3
        elif taskTarget == 2:
            lineColor = "2"
            calComplete = 4
        else:
            lineColor = "0"
            calComplete = 0

        # img = imageDealClass.getImgFromBytes(taskData)
        # cv2.namedWindow(str(taskTarget), cv2.WINDOW_NORMAL)
        # cv2.imshow(str(taskTarget), img)
        # cv2.waitKey(1)

        # print(imageDealClass.getTheLightColor(taskData))
        # out = imageDealClass.getTheLightColor(taskData)  # getTheLightColor
        # out = imageDealClass.getTheHuman(taskData)  # getTheHuman
        out = "no human"

        # 计算任务
        # print(out)
        # [24:32]human  [32:33]lineColor  [33:34]warning  [34:35]warningCarTarget  [35:39]suggestSpeed
        calResult_str = "{}".format(out + lineColor + str(warning) + str(AIM) + str(SPE))
        calResult_byte = calResult_str.encode('utf-8')
        # 组织计算结果
        return_data = calComplete.to_bytes(1, 'little') + len(calResult_byte).to_bytes(2, 'little') + calResult_byte
        shm.writeCalResult(ID, return_data)


def cameraAruco():
    global warning, AIM, SPE
    target1Point, target2Point = [], []
    mtx, dist, rMatrix, tvec, refMarkerArray, targetMarker = parameterPrepare()

    # vc = cv2.VideoCapture("./WIN.mp4")
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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


def test():
    global warning, AIM, SPE
    while True:
        print("111111111111111111111111111")
        # print(str(SPE)[0: 4])
        # if warning != 0:
        #     print("111111111111111111111111111    " + str(warning))
        # if AIM != 0:
        #     print("222222222222222222222222222")
        if SPE is not None:
            print("3333333333333333333333333333")
        # time.sleep(1)


def listenAndSend(listenPort, sendPort):
    imageDetect = taskImageDetect.taskImageDetest()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', listenPort))
    data, _ = s.recvfrom(1024)
    s.sendto(data, ('127.0.0.1', sendPort))
    numOfShm, idArray = getShmInfo(data)
    for i in range(numOfShm):
        # compTaskWith(idArray[i], imageDetect, lineColor)
        t = threading.Thread(target=compTaskWith(idArray[i], imageDetect),
                             name="LoopThreadWith{}".format(idArray[i]))
        # t = threading.Thread(target=compTaskWith, args=(idArray[i], imageDetect, lineColor))
        t.start()


def main():
    # t = threading.Thread(target=test, name="test")
    # t.start()

    ThreadAruco = threading.Thread(target=cameraAruco, name="cameraAruco")
    ThreadLAndS1 = threading.Thread(target=listenAndSend, args=(5214, 5215))
    # ThreadAruco.start()
    ThreadLAndS1.start()


if __name__ == '__main__':
    main()
