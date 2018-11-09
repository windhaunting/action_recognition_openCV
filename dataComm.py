#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:10:23 2018

@author: fubao
"""

#data, file related operation
import os
import cv2
import logging
from sys import stdout


def loggingSetting(outLogPath, logLevel):
    '''
    set logging level
    '''
    '''
    #logging.getLogger().setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logLevel)
    
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logLevel)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    fh = logging.FileHandler(outLogPath)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    '''
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handlers = [logging.FileHandler(outLogPath), logging.StreamHandler(stdout)]
    logging.basicConfig(level = logLevel, format = formatter, handlers = handlers)
    


def readVideo(inputVideoPath):
    cap = cv2.VideoCapture(inputVideoPath)      # 0 to camera in the file

    return cap



def extractVideoFrames(inputVideoPath, outFramesPath):
    '''
    extracframes from a video
    '''
    cap = readVideo(inputVideoPath)
    
    if (not cap.isOpened):
        print ('cam not opened: %s ', cap.isOpened())
        return 

    FPS = cap.get(cv2.CAP_PROP_FPS)
    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    NUMFRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    print('cam stat: ', FPS, WIDTH, HEIGHT, NUMFRAMES)
    
    count = 1

    while True:
      
      ret, img = cap.read()
      cv2.imwrite(os.path.join(outFramesPath, '%d.jpg') % count, img)     # save frame as JPEG file
      
      if not ret:
          print ("no frame exit here 1, total frames ")
          break

      count += 1
  
    
if __name__== "__main__":

    inputVideoPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/test1-ObjectDetection/inputOutputData/inputData/videos/cats/running/running_01/cat_running_01.mp4"
    outFramesPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/test1-ObjectDetection/inputOutputData/inputData/videos/cats/running/running_01/"
    extractVideoFrames(inputVideoPath, outFramesPath)

