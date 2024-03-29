#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:50:14 2018

@author: fubao
"""

#use object detection idea to identify object (human, basketballs, baskets) etc
# if they are both detected in a frame, then we judge basketball  center coordinate x is above hoop coordinate cetner x
# , and obove the x coordinate of human center coordinate



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
import logging

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


        
        
def extendSectionSize(x,y, w, h, ballSize, frameShape, extendRatio):
    '''
    extend region size around a detected region
    '''
    #EXTEND = 15
    xA = x-ballSize[0]*extendRatio if (x-ballSize[0]*extendRatio) > 0 else 0
    yA = y-ballSize[1]*extendRatio if (y-ballSize[1]*extendRatio) > 0 else 0
    
    xB = x+w+ballSize[0]*extendRatio if (x+w+ballSize[0]*extendRatio) < frameShape[1] else frameShape[1]        # frameshape[0] is height
    yB = y+h+ballSize[1]*extendRatio if (y+h+ballSize[1]*extendRatio) < frameShape[0]  else frameShape[0]
        
    return xA, yA, xB, yB



      
      
def detectBasketballDunkKFrameFixedWindow(videoPath, outputVideoName, fpsRed, reso, K, ratioKFrame):
    '''
    profiling of detect basketball dunk action 
    K: FixedWindow  every K frame together to decide an action;  if they are all detected   
    fpsRed:  fps reduced
    resolutionPx:  reduced resolution
    
    '''
    basketballModelPath = "../inputData/kinetics600/basketBallDetectionTrain/basketballTrainedModel/cascade.xml"
    humanModelPath = "../model/haarcascade_upperbody.xml"       # haarcascade_fullbody.xml   haarcascade_upperbody.xml
    basketHoopModelPath = "../inputData/kinetics600/basketballHoopTrain/basketballHoopTrainedModel/cascade.xml"
    

    basketBallDetectParameter = basketBallParameterCls(basketballModelPath, 1.15, 7, (5,5)) 
    humanDetectParameter = humanDetectParameterCls(humanModelPath, 1.05, 4, (10,10))
    basketHoopParameter = basketHoopParameterCls(basketHoopModelPath, 1.03, 1, (10,10))
    
    
    basketballCascade = cv2.CascadeClassifier(basketBallDetectParameter.modelPath)
    humanCascade = cv2.CascadeClassifier(humanDetectParameter.modelPath)
    basketHoopCascade = cv2.CascadeClassifier(basketHoopParameter.modelPath)
    
    print ("videoPath: ", videoPath)
    
    cap = readVideo(videoPath)
    
    #cap.set(cv2.CAP_PROP_FPS, fpsRed)  not working

    fps = cap.get(cv2.CAP_PROP_FPS)

    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    NUMFRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    

    #print ('cam stat: %s, %s, %s, %s ', fps, WIDTH, HEIGHT, NUMFRAMES)


    # outputVideoName =  "UCF101_v_longJump_g01_c01_out.avi"   # "UCF101_v_BasketballDunk_g01_c01_out.avi"  # "humanRunning_JHMDB_output_001.avi"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # X264 MP4V XVID MJPG
    outputVideoDir = os.path.join( os.path.dirname(__file__), '../output-Kinetics/')
    
    outputVideoDir += videoPath.split("/")[-2]
    if not os.path.exists(outputVideoDir):
        os.makedirs(outputVideoDir)

    finalOutDir = outputVideoDir + '/' + outputVideoName + "/"
    if not os.path.exists(finalOutDir):
        os.makedirs(finalOutDir)
    
    #outLogPath = finalOutDir + outputVideoName + "_log.txt"
    #logLevel = logging.WARNING 
    #logger = loggingSetting(outLogPath, logLevel)
    
    #logger.info('cam stat video name: %s, fps: %s, Wdith: %s, Height: %s, FRAMES: %s ', videoPath.split("/")[-1], fps, WIDTH, HEIGHT, NUMFRAMES)

    outVideo = cv2.VideoWriter(finalOutDir  + outputVideoName, fourcc, fps,  (int(WIDTH), int(HEIGHT)))
    
    actionDunkCnt = 0    # final action number
    actionDunkEachKFrameCnt = 0   # action dunk result in each frame in K frame ; 
    
    startFrm = 0  
    endFrm = 0
    realFrameNum = 0
    filterFrameNumMax = int(25//fpsRed-1)          # maximum is 25 sampling rate, filter frame in each second because of frame rate reduction ; extract every fps frame

    # exception processing
    if NUMFRAMES < 25 and fpsRed == 1:
        filterFrameNumMax = NUMFRAMES -1
        
    startTime = time.time()
    exitFlag = False
    while (startFrm < NUMFRAMES + 1 -K and not exitFlag):
        # Capture frame-by-frame
        
        endFrm = 0
        filterCnt = 0               # for fps sampling rate
        while (endFrm < K) and (startFrm < NUMFRAMES):
            
            ret, frame = cap.read()
            if not ret:
                print ("no frame exit here 1, total frames ", ret)
                exitFlag = True
                break

            if filterCnt < filterFrameNumMax:
                filterCnt += 1
                continue
            elif filterCnt >= filterFrameNumMax:
                filterCnt = 0
            
            # print ("realFrameNum frames ", NUMFRAMES, realFrameNum, filterCnt, filterFrameNumMax)
            
            # change resolution
            frame = cv2.resize(frame, reso);
            
            #imScale = cv2.resize(frame,(300,300))                    # Downscale to improve frame rate
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)            # gray shape (height, width)
            #print ("model pathssss: ", ret)
            
            #time.sleep(1)
                    
            # detect basketball first; then detect human around basketball
            balls, rejectLevelsBalls, levelWeightsBalls = basketballCascade.detectMultiScale3(
                gray,
                scaleFactor=basketBallDetectParameter.scaleFactor,
                minNeighbors=basketBallDetectParameter.minNeighbors,
                minSize=basketBallDetectParameter.minSize,
                flags=cv2.CASCADE_SCALE_IMAGE,
                outputRejectLevels=True
            )
            
            #print ("balls: ", type(balls), len(balls), gray.shape) 
            # Draw a rectangle around the faces
            baI = 0
            for (x, y, w, h) in balls:
                if levelWeightsBalls[baI][0] >= 0.5:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)                # GREEN  for basketball
                    # print ("xxxa : ",  x, y, gray.shape, baI, levelWeightsBalls, len(levelWeightsBalls))
                    cv2.putText(frame, str(round(levelWeightsBalls[baI][0], 3)), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)     # Text in black "Ball" 
                    
                    baI += 1
                    # get basketball's around area  how big
                    ballSize = (w//2, h//2)
                    
                    xA, yA, xB, yB = extendSectionSize(x, y, w, h, ballSize, gray.shape, 30)
                    
                    cropImg_DetectHuman = frame[yA:yB, xA:xB]  
                    
                    #cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 0, 0), 3)            # for basketball's extended region
                    
                    # detct human inside cropImg_DetectHuman
                    humanGray = cv2.cvtColor(cropImg_DetectHuman, cv2.COLOR_BGR2GRAY)
                    
                    # print ("humanGray: ", len(humanGray), humanGray.shape)
                    humans, rejectLevelsHumans, levelWeightsHumans = humanCascade.detectMultiScale3(
                        humanGray,
                        scaleFactor=humanDetectParameter.scaleFactor,
                        minNeighbors=humanDetectParameter.minNeighbors,
                        minSize=humanDetectParameter.minSize,
                        flags=cv2.CASCADE_SCALE_IMAGE,
                        outputRejectLevels = True
                    )
                    
                    #print ("humans: ", type(balls), len(balls), humanGray.shape)
                    hmI = 0
                    for (humX, humY, humW, humH) in humans:
                        if levelWeightsHumans[hmI][0] >= 0.5:
                            originFrameX = humX + x if xA != 0 else humX            # humX +x
                            originFrameY = humY + y if yA != 0 else humY            # # humX +y
                            cv2.rectangle(frame, (originFrameX, originFrameY), (originFrameX+humW, originFrameY+humH), (0, 0, 255), 3)      #RED for human
                            #cv2.rectangle(frame, (humX, humY), (humX+humW, humY+humH), (0, 0, 255), 3)      #RED for human
                            cv2.putText(frame, str(round(levelWeightsHumans[hmI][0], 3)), (originFrameX, originFrameY), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)     # Text in black "Human"
        
                            # also detect basketball hoop
                            # detct human inside cropImg_DetectHuman
                            hmI += 1
                        
                            xA, yA, xB, yB = extendSectionSize(x, y, w, h, ballSize, gray.shape, 15)
                
                            cropImg_DetectHoop = frame[yA:yB, xA:xB]  
                            #print ("xxx : ",  x, y, ballSize, gray.shape)
                            #print (" xA, yA, xB, yB : ",  xA, yA, xB, yB, gray.shape)
                            hoopGray = cv2.cvtColor(cropImg_DetectHoop, cv2.COLOR_BGR2GRAY)
                            
                            hoops, rejectLevelsHoops, levelWeightsHoops = basketHoopCascade.detectMultiScale3(
                                hoopGray,
                                scaleFactor=basketHoopParameter.scaleFactor,
                                minNeighbors=basketHoopParameter.minNeighbors,
                                minSize=basketHoopParameter.minSize,
                                flags=cv2.CASCADE_SCALE_IMAGE,
                                outputRejectLevels=True
                            )
                            
                            
                            #print ("hoops: ", type(hoops), len(hoops), hoopGray.shape)
                            hpI = 0
                            for (hoopX, hoopY, hoopW, hoopH) in hoops:
                                if levelWeightsHoops[hpI][0] >= 0.1:
                                    originFrameX = hoopX + x if xA != 0 else hoopX
                                    originFrameY = hoopY + y if yA != 0 else hoopY 
                                    cv2.rectangle(frame, (originFrameX, originFrameY), (originFrameX+hoopW, originFrameY+hoopH), (0, 255, 255), 3)       # yellow for basketball hoop
                                    
                                    #cv2.rectangle(frame, (hoopX, hoopY), (hoopX+hoopW, hoopY+hoopH), (0, 255, 255), 3) # yellow for basketball hoop
                                    cv2.putText(frame, str(round(levelWeightsHoops[hpI][0], 3)), (originFrameX, originFrameY), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)     # Text in black "Hoop"
                                    
                                    hpI += 1
                                    
                                    # decide the action of basketball dunk
                                    ballCenter = (x + w/2, y + h/2)
                                    
                                    for (hoopX, hoopY, hoopW, hoopH) in hoops:
                                        hoopCenter = (hoopX + hoopW/2, hoopY + hoopH/2)
                                        for (humX, humY, humW, humH) in humans:
                                            humanCenter = (humX + humW/2, humY + humH/2)
                                            #if ballCenter[1] > hoopCenter[1] and  hoopCenter[1] > humanCenter[1]:
                                            if x >= hoopX and y >= hoopY and (x+w) <= (hoopX+hoopW) and (y+h) <= (hoopY+hoopH) and hoopCenter[1] > humanCenter[1]:
                                                actionDunkEachKFrameCnt +=1
                                                #print ("actionDunk detected HERE HERE: ", actionDunkEachKFrameCnt)
                    
                                                if actionDunkEachKFrameCnt >= K*ratioKFrame: 
                                                    actionDunkCnt += 1
                                
            # Display the resulting frame
            #cv2.imshow('Video', frame)
            #outVideo.write(frame)
            realFrameNum += 1

            # change frame rate
            #cv2.waitKey( 1000 // fpsRed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
            startFrm += 1
            endFrm += 1
            
        # next loop;  clear last K frame detection result
        actionDunkEachKFrameCnt = 0 
        
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    outVideo.release()
    print ("actionDunk count: ", actionDunkCnt, fps, realFrameNum, NUMFRAMES)
    
    elapsedTime = time.time() - startTime
    #logger.info("actionDunk count %s ;  elapsedTime: %s s", actionDunkCnt, elapsedTime)
    
    
    return actionDunkCnt, elapsedTime, realFrameNum, NUMFRAMES


def detectBasketballDunkKFrameCVV(kFrameDict, ratioKFrame):
    '''
    K frame to detect the basketball dunk action existing or not
    input: k frames dict;  dict[0], dict[1].... dict[K] = frame matrix
           ratioKFrame: how many frames/K are effective for counting
    '''
    K = len(kFrameDict)
     
    basketballModelPath = "../inputData/kinetics600/basketBallDetectionTrain/basketballTrainedModel/cascade.xml"
    humanModelPath = "../model/haarcascade_upperbody.xml"       # haarcascade_fullbody.xml   haarcascade_upperbody.xml
    basketHoopModelPath = "../inputData/kinetics600/basketballHoopTrain/basketballHoopTrainedModel/cascade.xml"
    

    basketBallDetectParameter = basketBallParameterCls(basketballModelPath, 1.15, 7, (5,5)) 
    humanDetectParameter = humanDetectParameterCls(humanModelPath, 1.05, 4, (10,10))
    basketHoopParameter = basketHoopParameterCls(basketHoopModelPath, 1.03, 1, (10,10))
    
    
    basketballCascade = cv2.CascadeClassifier(basketBallDetectParameter.modelPath)
    humanCascade = cv2.CascadeClassifier(humanDetectParameter.modelPath)
    basketHoopCascade = cv2.CascadeClassifier(basketHoopParameter.modelPath)
    
    actionDunkEachKFrameCnt = 0
    actionDunkCnt = 0
    
    startTime = time.time()
    for cnt, frame in kFrameDict:
        x = 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)            # gray shape (height, width)
        #print ("model pathssss: ", ret)
        
        #time.sleep(1)
                
        # detect basketball first; then detect human around basketball
        balls, rejectLevelsBalls, levelWeightsBalls = basketballCascade.detectMultiScale3(
            gray,
            scaleFactor=basketBallDetectParameter.scaleFactor,
            minNeighbors=basketBallDetectParameter.minNeighbors,
            minSize=basketBallDetectParameter.minSize,
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True
        )
        
        
         #print ("balls: ", type(balls), len(balls), gray.shape) 
        # Draw a rectangle around the faces
        baI = 0
        for (x, y, w, h) in balls:
            if levelWeightsBalls[baI][0] >= 0.5:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)                # GREEN  for basketball
                # print ("xxxa : ",  x, y, gray.shape, baI, levelWeightsBalls, len(levelWeightsBalls))
                cv2.putText(frame, str(round(levelWeightsBalls[baI][0], 3)), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)     # Text in black "Ball" 
                
                baI += 1
                # get basketball's around area  how big
                ballSize = (w//2, h//2)
                
                xA, yA, xB, yB = extendSectionSize(x, y, w, h, ballSize, gray.shape, 30)
                
                cropImg_DetectHuman = frame[yA:yB, xA:xB]  
                
                #cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 0, 0), 3)            # for basketball's extended region
                
                # detct human inside cropImg_DetectHuman
                humanGray = cv2.cvtColor(cropImg_DetectHuman, cv2.COLOR_BGR2GRAY)
                
                # print ("humanGray: ", len(humanGray), humanGray.shape)
                humans, rejectLevelsHumans, levelWeightsHumans = humanCascade.detectMultiScale3(
                    humanGray,
                    scaleFactor=humanDetectParameter.scaleFactor,
                    minNeighbors=humanDetectParameter.minNeighbors,
                    minSize=humanDetectParameter.minSize,
                    flags=cv2.CASCADE_SCALE_IMAGE,
                    outputRejectLevels = True
                )
                
                #print ("humans: ", type(balls), len(balls), humanGray.shape)
                hmI = 0
                for (humX, humY, humW, humH) in humans:
                    if levelWeightsHumans[hmI][0] >= 0.5:
                        originFrameX = humX + x if xA != 0 else humX            # humX +x
                        originFrameY = humY + y if yA != 0 else humY            # # humX +y
                        cv2.rectangle(frame, (originFrameX, originFrameY), (originFrameX+humW, originFrameY+humH), (0, 0, 255), 3)      #RED for human
                        #cv2.rectangle(frame, (humX, humY), (humX+humW, humY+humH), (0, 0, 255), 3)      #RED for human
                        cv2.putText(frame, str(round(levelWeightsHumans[hmI][0], 3)), (originFrameX, originFrameY), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)     # Text in black "Human"
    
                        # also detect basketball hoop
                        # detct human inside cropImg_DetectHuman
                        hmI += 1
                    
                        xA, yA, xB, yB = extendSectionSize(x, y, w, h, ballSize, gray.shape, 15)
            
                        cropImg_DetectHoop = frame[yA:yB, xA:xB]  
                        #print ("xxx : ",  x, y, ballSize, gray.shape)
                        #print (" xA, yA, xB, yB : ",  xA, yA, xB, yB, gray.shape)
                        hoopGray = cv2.cvtColor(cropImg_DetectHoop, cv2.COLOR_BGR2GRAY)
                        
                        hoops, rejectLevelsHoops, levelWeightsHoops = basketHoopCascade.detectMultiScale3(
                            hoopGray,
                            scaleFactor=basketHoopParameter.scaleFactor,
                            minNeighbors=basketHoopParameter.minNeighbors,
                            minSize=basketHoopParameter.minSize,
                            flags=cv2.CASCADE_SCALE_IMAGE,
                            outputRejectLevels=True
                        )
                        
                        
                        #print ("hoops: ", type(hoops), len(hoops), hoopGray.shape)
                        hpI = 0
                        for (hoopX, hoopY, hoopW, hoopH) in hoops:
                            if levelWeightsHoops[hpI][0] >= 0.1:
                                originFrameX = hoopX + x if xA != 0 else hoopX
                                originFrameY = hoopY + y if yA != 0 else hoopY 
                                cv2.rectangle(frame, (originFrameX, originFrameY), (originFrameX+hoopW, originFrameY+hoopH), (0, 255, 255), 3)       # yellow for basketball hoop
                                
                                #cv2.rectangle(frame, (hoopX, hoopY), (hoopX+hoopW, hoopY+hoopH), (0, 255, 255), 3) # yellow for basketball hoop
                                cv2.putText(frame, str(round(levelWeightsHoops[hpI][0], 3)), (originFrameX, originFrameY), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)     # Text in black "Hoop"
                                
                                hpI += 1
                                
                                # decide the action of basketball dunk
                                #ballCenter = (x + w/2, y + h/2)
                                
                                for (hoopX, hoopY, hoopW, hoopH) in hoops:
                                    hoopCenter = (hoopX + hoopW/2, hoopY + hoopH/2)
                                    for (humX, humY, humW, humH) in humans:
                                        humanCenter = (humX + humW/2, humY + humH/2)
                                        #if ballCenter[1] > hoopCenter[1] and  hoopCenter[1] > humanCenter[1]:
                                        if x >= hoopX and y >= hoopY and (x+w) <= (hoopX+hoopW) and (y+h) <= (hoopY+hoopH) and hoopCenter[1] > humanCenter[1]:
                                            actionDunkEachKFrameCnt +=1
                                            #print ("actionDunk detected HERE HERE: ", actionDunkEachKFrameCnt)
                
                                            if actionDunkEachKFrameCnt >= K*ratioKFrame: 
                                                actionDunkCnt += 1
    
    print ("actionDunk count: ", actionDunkCnt)
    
    elapsedTime = time.time() - startTime
    #logger.info("actionDunk count %s ;  elapsedTime: %s s", actionDunkCnt, elapsedTime)
    
    
    return actionDunkCnt, elapsedTime
    
    
def detectBasketballDunk(videoPath, outputVideoName, fpsRed, reso):
    '''
    detect basketball dunk action every each frame
    
    fpsRed:  fps reduced
    resolutionPx:  reduced resolution
    
    '''
    basketballModelPath = "../inputData/kinetics600/basketBallDetectionTrain/basketballTrainedModel/cascade.xml"
    humanModelPath = "../model/haarcascade_fullbody.xml"       # haarcascade_fullbody.xml   haarcascade_upperbody.xml
    basketHoopModelPath = "../inputData/kinetics600/basketballHoopTrain/basketballHoopTrainedModel/cascade.xml"
    

    basketBallDetectParameter = basketBallParameterCls(basketballModelPath, 1.3, 10, (5,5)) 
    humanDetectParameter = humanDetectParameterCls(humanModelPath, 1.05, 3, (5,5))
    basketHoopParameter = basketHoopParameterCls(basketHoopModelPath, 1.05, 2, (5,5))
    
    
    basketballCascade = cv2.CascadeClassifier(basketBallDetectParameter.modelPath)
    humanCascade = cv2.CascadeClassifier(humanDetectParameter.modelPath)
    basketHoopCascade = cv2.CascadeClassifier(basketHoopParameter.modelPath)
    
    print ("videoPath: ", videoPath)
    
    cap = readVideo(videoPath)
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    NUMFRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    #print ('cam stat: %s, %s, %s, %s ', fps, WIDTH, HEIGHT, NUMFRAMES)
    


    # outputVideoName =  "UCF101_v_longJump_g01_c01_out.avi"   # "UCF101_v_BasketballDunk_g01_c01_out.avi"  # "humanRunning_JHMDB_output_001.avi"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # X264 MP4V XVID MJPG
    outputVideoDir = os.path.join( os.path.dirname(__file__), '../output-Kinetics/')
   
    finalOutDir = outputVideoDir + outputVideoName + "/"
    if not os.path.exists(finalOutDir):
        os.makedirs(finalOutDir)
    
    #outLogPath = finalOutDir + outputVideoName + "_log.txt"
    #logLevel = logging.WARNING 
    #logger = loggingSetting(outLogPath, logLevel)
    
    #logger.info('cam stat video name: %s, fps: %s, Wdith: %s, Height: %s, FRAMES: %s ', videoPath.split("/")[-1], fps, WIDTH, HEIGHT, NUMFRAMES)

    outVideo = cv2.VideoWriter(finalOutDir  + outputVideoName, fourcc, fps,  (int(WIDTH), int(HEIGHT)))
    
    cntFrm = 0
    actionDunkCnt = 0
    
    startTime = time.time()
    while True:
        # Capture frame-by-frame
         
        ret, frame = cap.read()
        
        if not ret:
            print ("no frame exit here 1, total frames ", ret)
            break
        
        # change resolution
        frame = cv2.resize(frame, reso);
        
        #imScale = cv2.resize(frame,(300,300)) # Downscale to improve frame rate
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)            # gray shape (height, width)
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
    
        #print ("balls: ", type(balls), len(balls), gray.shape) 
        # Draw a rectangle around the faces
        for (x, y, w, h) in balls:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)         # GREEN  for basketball
            #print ("xxxa : ",  x, y, gray.shape)
            # get basketball's around area  how big
            ballSize = (w//2, h//2)
            
            xA, yA, xB, yB = extendSectionSize(x, y, w, h, ballSize, gray.shape, 20)
            
            cropImg_DetectHuman = frame[yA:yB, xA:xB]  
            
            #cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 0, 0), 3)            # for basketball's extended region
            
            # detct human inside cropImg_DetectHuman
            humanGray = cv2.cvtColor(cropImg_DetectHuman, cv2.COLOR_BGR2GRAY)
            
            #print ("humanGray: ", len(humanGray), humanGray.shape)
            humans = humanCascade.detectMultiScale(
                humanGray,
                scaleFactor=humanDetectParameter.scaleFactor,
                minNeighbors=humanDetectParameter.minNeighbors,
                minSize=humanDetectParameter.minSize,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            #print ("humans: ", type(balls), len(balls), humanGray.shape)
            for (humX, humY, humW, humH) in humans:
                originFrameX = humX + x if xA != 0 else humX          # humX +x
                originFrameY = humY + y if yA != 0 else humY            # # humX +y
                
                cv2.rectangle(frame, (originFrameX, originFrameY), (originFrameX+humW, originFrameY+humH), (0, 0, 255), 3)      #RED for human
                
            #also detect basketball hoop
             # detct human inside cropImg_DetectHuman
             
            xA, yA, xB, yB = extendSectionSize(x, y, w, h, ballSize, gray.shape, 15)

            
            cropImg_DetectHoop = frame[yA:yB, xA:xB]  
            #print ("xxx : ",  x, y, ballSize, gray.shape)
            #print (" xA, yA, xB, yB : ",  xA, yA, xB, yB, gray.shape)
            hoopGray = cv2.cvtColor(cropImg_DetectHoop, cv2.COLOR_BGR2GRAY)
            
            hoops = basketHoopCascade.detectMultiScale(
                hoopGray,
                scaleFactor=basketHoopParameter.scaleFactor,
                minNeighbors=basketHoopParameter.minNeighbors,
                minSize=basketHoopParameter.minSize,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            #print ("hoops: ", type(hoops), len(hoops), hoopGray.shape)
            for (hoopX, hoopY, hoopW, hoopH) in hoops:
                originFrameX = hoopX + x if xA != 0 else hoopX
                originFrameY = hoopY + y if yA != 0 else hoopY 
                cv2.rectangle(frame, (originFrameX, originFrameY), (originFrameX+hoopW, originFrameY+hoopH), (0, 255, 255), 3) # yellow for basketball hoop
                
            
            #decide the action of basketball dunk
            ballCenter = (x + w/2, y + h/2)
            
            for (hoopX, hoopY, hoopW, hoopH) in hoops:
                hoopCenter = (hoopX + hoopW/2, hoopY + hoopH/2)
                for (humX, humY, humW, humH) in humans:
                    humanCenter = (humX + humW/2, humY + humH/2)
                    if ballCenter[1] > hoopCenter[1] and  hoopCenter[1] > humanCenter[1]:
                        print ("actionDunk detected HERE HERE Frame: ", cntFrm)
                        actionDunkCnt += 1
                            
        # Display the resulting frame
        #cv2.imshow('Video', frame)
        #outVideo.write(frame)
        
        # change frame rate
        cv2.waitKey( 1000 // fpsRed)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cntFrm += 1
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    outVideo.release()
    print ("actionDunk count: ", actionDunkCnt)
    elapsedTime = time.time() - startTime
    #logger.info("actionDunk count %s ;  elapsedTime: %s s", actionDunkCnt, elapsedTime)
    
    
    return actionDunkCnt, elapsedTime, NUMFRAMES


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