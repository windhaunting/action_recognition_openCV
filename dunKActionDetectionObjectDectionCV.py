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


# define model path and parameter for each object detection
class humanDetectParameterCls:
    def __init__(self, modelPath, scaleFactor, minNeighbors, minSize):
        self.modelPath = modelPath   #  ""
        self.scaleFactor =  scaleFactor   # 1.05
        self.minNeighbors = minNeighbors   # 3
        self.minSize =  minSize          #(10, 10)
    
class basketBallParameterCls:
    def __init__(self, modelPath, scaleFactor, minNeighbors, minSize):
        self.modelPath = modelPath   #  ""
        self.scaleFactor =  scaleFactor   # 1.05
        self.minNeighbors = minNeighbors   # 3
        self.minSize =  minSize          #(10, 10)
        
    
class basketHoopParameterCls:
    def __init__(self, modelPath, scaleFactor, minNeighbors, minSize):
        self.modelPath = modelPath   #  ""
        self.scaleFactor =  scaleFactor   # 1.05
        self.minNeighbors = minNeighbors   # 3
        self.minSize =  minSize          #(10, 10)
        
def extendSectionSize(x,y, w, h):
    '''
    extend region sizei
    '''
    EXTEND = 15
    xA = x-ballSize[0]*EXTEND if (x-ballSize[0]*EXTEND) > 0 else 0
    yA = y-ballSize[1]*EXTEND if (x-ballSize[1]*EXTEND) > 0 else 0
    
    xB = x+w+ballSize[0]*EXTEND if (x+w+ballSize[0]*EXTEND) < gray.shape[0] else gray.shape[0]
    
    yB = y+h+ballSize[1]*EXTEND if (y+h+ballSize[1]*EXTEND) > gray.shape[1]  else gray.shape[1]
        
    return xA, yA, xB, yB

def detectBasketballDunk(videoPath, outputVideoName):
    '''
    detect basketball dunk action
    
    '''
    basketballModelPath = "../inputData/kinetics600/basketBallDetectionTrain/basketballTrainedModel/cascade.xml"
    humanModelPath = "/home/fubao/program/opencv/data/haarcascades/haarcascade_fullbody.xml"
    basketHoopModelPath = "../inputData/kinetics600/basketballHoopTrain/basketballHoopTrainedModel/cascade.xml"
    

    basketBallDetectParameter = basketBallParameterCls(basketballModelPath, 1.3, 5, (5,5)) 
    humanDetectParameter = humanDetectParameterCls(humanModelPath, 1.05, 3, (10,10))
    basketHoopParameter = basketHoopParameterCls(basketHoopModelPath, 1.05, 2, (5,5))
    
    
    basketballCascade = cv2.CascadeClassifier(basketBallDetectParameter.modelPath)
    humanCascade = cv2.CascadeClassifier(humanDetectParameter.modelPath)
    basketHoopCascade = cv2.CascadeClassifier(basketHoopParameter.modelPath)
    
    print ("videoPath, model path: ", videoPath, )
    
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
    
    outVideo = cv2.VideoWriter(finalOutDir  + outputVideoName, fourcc, 10,  (int(WIDTH), int(HEIGHT)))
    
    
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
                
        # detect basketball first; then detect human around basketball
        balls = basketballCascade.detectMultiScale(
            gray,
            scaleFactor=basketBallDetectParameter.scaleFactor,
            minNeighbors=basketBallDetectParameter.minNeighbors,
            minSize=basketBallDetectParameter.minSize,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
        print ("balls: ", type(balls), len(balls), gray.shape)
        # Draw a rectangle around the faces
        for (x, y, w, h) in balls:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # get basketball's around area  how big
            ballSize = (w//2, h//2)

            EXTEND = 14
            xA = x-ballSize[0]*EXTEND if (x-ballSize[0]*EXTEND) > 0 else 0
            yA = y-ballSize[1]*EXTEND if (x-ballSize[1]*EXTEND) > 0 else 0
            
            xB = x+w+ballSize[0]*EXTEND if (x+w+ballSize[0]*EXTEND) < gray.shape[0] else gray.shape[0]
            
            yB = y+h+ballSize[1]*EXTEND if (y+h+ballSize[1]*EXTEND) > gray.shape[1]  else gray.shape[1]

            cropImg_DetectHuman = frame[yA:yB, xA:xB]  
            
            cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 0, 0), 3)
            
            # detct human inside cropImg_DetectHuman
            humanGray = cv2.cvtColor(cropImg_DetectHuman, cv2.COLOR_BGR2GRAY)
            
            humans = humanCascade.detectMultiScale(
                humanGray,
                scaleFactor=humanDetectParameter.scaleFactor,
                minNeighbors=humanDetectParameter.minNeighbors,
                minSize=humanDetectParameter.minSize,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print ("humans: ", type(balls), len(balls), humanGray.shape)
            for (humX, humY, humW, humH) in humans:
                originFrameX = humX + x if xA != 0 else humX          # humX +x
                originFrameY = humY + y if yA != 0 else humY            # # humX +y
                cv2.rectangle(frame, (originFrameX, originFrameY), (originFrameX+humW, originFrameY+humH), (0, 0, 255), 3)
                
            #also detect basketball hoop
             # detct human inside cropImg_DetectHuman
            EXTEND = 15
            xA = x-ballSize[0]*EXTEND if (x-ballSize[0]*EXTEND) > 0 else 0
            yA = y-ballSize[1]*EXTEND if (x-ballSize[1]*EXTEND) > 0 else 0
            
            xB = x+w+ballSize[0]*EXTEND if (x+w+ballSize[0]*EXTEND) < gray.shape[0] else gray.shape[0]
            
            yB = y+h+ballSize[1]*EXTEND if (y+h+ballSize[1]*EXTEND) > gray.shape[1]  else gray.shape[1]

            cropImg_DetectHuman = frame[yA:yB, xA:xB]  
            
            hoopGray = cv2.cvtColor(cropImg_DetectHuman, cv2.COLOR_BGR2GRAY)
            
            hoops = basketHoopCascade.detectMultiScale(
                hoopGray,
                scaleFactor=basketHoopParameter.scaleFactor,
                minNeighbors=basketHoopParameter.minNeighbors,
                minSize=basketHoopParameter.minSize,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print ("hoops: ", type(hoops), len(hoops), hoopGray.shape)
            for (hoopX, hoopY, hoopW, hoopH) in hoops:
                originFrameX = hoopX + x if xA != 0 else hoopX
                originFrameY = hoopY + y if yA != 0 else hoopY 
                cv2.rectangle(frame, (originFrameX, originFrameY), (originFrameX+hoopW, originFrameY+hoopH), (0, 255, 255), 3)
            
            
            #decide
            
        # Display the resulting frame
        cv2.imshow('Video', frame)
        outVideo.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    outVideo.release()
    
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
    
    outVideo = cv2.VideoWriter(finalOutDir  + outputVideoName, fourcc, 10,  (int(WIDTH), int(HEIGHT)))
    
    
    
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
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
        print ("objs: ", type(objs), len(objs))
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
    
    '''
    #modelPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/inputData/kinetics600/basketBallDetectionTrain/basketballTrainedModel/cascade.xml"
    
    modelPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/inputData/kinetics600/basketballHoopTrain/basketballHoopTrainedModel/cascade.xml"
    
    #videoPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/inputData/kinetics600/videos-dunkBasketball/dunking_basketball/oyBZJZdiCQk_000007_000017.mp4"
    videoPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/inputData/kinetics600/videos-dunkBasketball/dunking_basketball/zxVjaOAmDBk_000000_000010.mp4"

    
    #videoPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/inputData/kinetics600/videos-dunkBasketball/dunking_basketball/Zrm9_xE85d8_000001_000011.mp4"
    #videoPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/inputData/kinetics600/videos-dunkBasketball/dunking_basketball/0boDimB7PrA_000002_000012.mp4"

    #videoPath = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/inputData/kinetics600/videos-dunkBasketball/dunking_basketball/0A8ov5xMuAM_000003_000013.mp4"

    #outputVideoName = "oyBZJZdiCQk_000007_000017_basketballDetect_out.mp4"
    outputVideoName = videoPath.split("/")[-1] + "_basketballHoopDetect_out.mp4"     #_basketballDetect_out  _basketballHoopDetect_out

    detectBasketBall(modelPath, videoPath, outputVideoName)
    
    '''
    exec(sys.argv[1])