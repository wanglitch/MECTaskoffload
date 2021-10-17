# coding=utf-8
# ! /home/magizio/miniconda3/bin/python
import time

import sysv_ipc as ipc

# byte 转 int
# int 转 byte
# |1:1:4:n|
# |发送任务小车:任务类型:任务长度:任务数据|


def getShmInfo(data):
    dataStr = data.decode("utf-8")
    data_split = dataStr.split("|:|:|")
    numOfShm = data[0]
    id_array = []
    print(data)
    for i in range(numOfShm):
        id_array.append(int(data_split[i + 1]))
    # print(id_array)
    return numOfShm, id_array


# index:从0开始计数
def getByte(ID, index):
    shm = ipc.attach(id=ID)
    byte = shm.read(index + 1)[index]
    # print(byte)
    # time.sleep(0.6)
    shm.detach()
    return byte


def isaTaskWaitingToDeal(ID):
    # if getByte(ID, 0) == 4:
    #     return True
    # else:
    #     return False
    if getByte(ID, 0) == 1:
        return 1
    elif getByte(ID, 0) == 2:
        return 2
    else:
        return -1


def writeCalResult(ID, data):
    shm = ipc.attach(id=ID)
    shm.write(data)
    shm.detach()


def getTheTask(ID):
    shm = ipc.attach(id=ID)
    byteALL = shm.read()
    taskTarget = byteALL[0]
    taskType = byteALL[1]
    taskLength = int.from_bytes(byteALL[2:6], 'big')
    taskData = byteALL[6:6 + taskLength]
    # taskData = byteALL[1:]
    shm.detach()
    return taskTarget, taskType, taskLength, taskData


def getTheTaskIfThereIsOne(ID):
    # while isaTaskWaitingToDeal(ID) is False: pass
    while isaTaskWaitingToDeal(ID) == -1:
        pass
    taskTarget, taskType, taskLength, taskData = getTheTask(ID)
    return taskTarget, taskType, taskLength, taskData
