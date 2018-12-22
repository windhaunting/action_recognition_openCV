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
from collections import defaultdict


def loggingSetting(outLogPath, logLevel):
    '''
    set logging level
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
        
    # create a file handler
    handler = logging.FileHandler(outLogPath)
    handler.setLevel(logLevel)
    
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(handler)

    return logger
    


def readVideo(inputVideoPath):
    '''
    read a video
    '''
    
    cap = cv2.VideoCapture(inputVideoPath)      # 0 to camera in the file

    return cap



def extractVideoFrames(inputVideoPath, outFramesPath, saveFileOrDict):
    '''
    extracframes from a video
    and save into file or dictionary
    
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
    imageDict = defaultdict(int)
    while True:
      
      ret, img = cap.read()

          
      if not ret:
          print ("no frame exit here 1, total frames ")
          break
      
      # test resize resolution
      
      img = cv2.resize(img, (300, 300));
      if saveFileOrDict == "file":
          cv2.imwrite(os.path.join(outFramesPath, '%d.jpg') % count, img)     # save frame as JPEG file
      elif saveFileOrDict == "dict":
          imageDict[count] = img



      count += 1
  
      #cv2.waitKey( 1000 // 100)
    if saveFileOrDict == "file":
        return None
    elif saveFileOrDict == "dict":
        return imageDict
    
 
if __name__== "__main__":

    inputVideoPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/test1-ObjectDetection/inputOutputData/inputData/videos/cats/running/running_01/cat_running_01.mp4"
    outFramesPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/test1-ObjectDetection/inputOutputData/inputData/videos/cats/running/running_011/"
    extractVideoFrames(inputVideoPath, outFramesPath, "file")

