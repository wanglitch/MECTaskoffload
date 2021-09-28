#!/usr/bin/python
# coding=utf-8

import sys

# import keyboardimport sys
sys.path.append("/home/wan/MEC intelligent/Intelligent-Traffic-Based-On-CV")
sys.path.append("/home/wan/MEC intelligent/Intelligent-Traffic-Based-On-CV/scripts")

import os
import cv2
import numpy as np
from targetdetect import targetdetect


# target_detect = targetdetect()


class taskImageDetest:
    def __init__(self):
        self.targetdetect = targetdetect()

    def getImgFromBytes(self, rawBin):
        nparr = np.fromstring(rawBin, np.uint8)
        img = cv2.imdecode(nparr, 1)
        # cv2.imshow('1', img)
        # cv2.waitKey(1)
        return img

    def getTheLightColor(self, rawBin):
        img = self.getImgFromBytes(rawBin)
        return self.targetdetect.trafficLightDetect(img)

    def getTheHuman(self, rawBin):
        img = self.getImgFromBytes(rawBin)
        return self.targetdetect.humanDetect(img)
