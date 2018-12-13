#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:50:14 2018

@author: fubao
"""

#use object detection idea to identify object (human, basketballs, baskets) etc


# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
import sys
import os
import imutils
import math
import time

from collections import deque

from dataComm import loggingSetting
from dataComm import readVideo



def detectBasketBall(modelPath, videoPath, outputVideoName):

    '''
    xml model pretrained in folder basketballTrainingRecognition
    videoPath: input a vdideo
    
    '''
    
    objectCascade = cv2.CascadeClassifier(modelPath)
    print ("videoPath, model path: ", videoPath, modelPath)
    
    cap = readVideo(videoPath)
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    NUMFRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    print ('cam stat: %s, %s, %s, %s ', fps, WIDTH, HEIGHT, NUMFRAMES)
    
    # outputVideoName =  "UCF101_v_longJump_g01_c01_out.avi"   # "UCF101_v_BasketballDunk_g01_c01_out.avi"  # "humanRunning_JHMDB_output_001.avi"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # X264 MP4V XVID MJPG
    outputVideoDir = os.path.join( os.path.dirname(__file__), '../output-Kinetics/')
   
    finalOutDir = outputVideoDir + outputVideoName + "/"
    if not os.path.exists(finalOutDir):
        os.makedirs(finalOutDir)
    
    outVideo = cv2.VideoWriter(finalOutDir  + outputVideoName, fourcc, 5,  (int(WIDTH), int(HEIGHT)))
    
    
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print ("no frame exit here 1, total frames ", ret)
            break
        
        #imScale = cv2.resize(frame,(300,300)) # Downscale to improve frame rate
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print ("model pathssss: ", ret)
        
        #time.sleep(1)
                
        objs = objectCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
        # Draw a rectangle around the faces
        for (x, y, w, h) in objs:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # Display the resulting frame
        cv2.imshow('Video', frame)
        outVideo.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    outVideo.release()
    
    
if __name__== "__main__":
    
    modelPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/inputData/kinetics600/basketBallDetectionTrain/basketballTrainedModel/cascade.xml"
    videoPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/inputData/kinetics600/videos-dunkBasketball/dunking basketball/oyBZJZdiCQk_000007_000017.mp4"
    outputVideoName = "oyBZJZdiCQk_000007_000017_basketballDetect_out.mp4"
    detectBasketBall(modelPath, videoPath, outputVideoName)
