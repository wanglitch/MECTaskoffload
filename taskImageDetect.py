#!/usr/bin/python
# coding=utf-8
import sys
import os
import cv2
import numpy as np
from targetdetect import TargetDetect
# import keyboardimport sys
sys.path.append("/home/wan/MEC intelligent/Intelligent-Traffic-Based-On-CV")
sys.path.append("/home/wan/MEC intelligent/Intelligent-Traffic-Based-On-CV/scripts")


class TaskImageDetect:
    def __init__(self):
        self.targetDetect = TargetDetect()

    def getImgFromBytes(self, rawBin):
        npArr = np.fromstring(rawBin, np.uint8)
        img = cv2.imdecode(npArr, 1)
        # cv2.imshow('1', img)
        # cv2.waitKey(1)
        return img

    def getTheLightColor(self, rawBin):
        img = self.getImgFromBytes(rawBin)
        return self.targetDetect.trafficLightDetect(img)

    def getTheHuman(self, rawBin):
        img = self.getImgFromBytes(rawBin)
        return self.targetDetect.humanDetect(img)
