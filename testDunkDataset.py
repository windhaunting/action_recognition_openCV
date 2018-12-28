#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:00:56 2018

@author: fubao
"""

# test the effecct of tradtional opencv on basketball dunk action
# in a systematical way
from dunKActionDetectionObjectDectionCV import detectBasketballDunk
from dunKActionDetectionObjectDectionCV import detectBasketballDunkKFrameFixedWindow

import logging
import glob
import sys
import os 
import csv
import matplotlib.pyplot as plt

from dataComm import loggingSetting


def testAllVideosDir():
        
    K = 6     # each K frames
    ratioKFrame = 0.1    # how many frames detected as dunk over K frame
    frameRates = [30, 10, 5, 2, 1]    # [30]  #  [30, 10, 5, 2, 1]    # [30],  [30, 10, 5, 2, 1] 
    resoPixels =  [720, 600, 480, 360, 240]   # [720]     # [720, 600, 480, 360, 240]  #  [720]    # [720, 600, 480, 360, 240]            #  16: 9
    resolutions = [(w*16//9, w) for w in resoPixels]
    inputVideoDir = "../inputData/kinetics600/videos-dunkBasketball/testVideo01_trimmed_10videos/"
    # get video from inputVideoDir
    filePaths = glob.glob(inputVideoDir + "*.mp4")
    #print ("files: ", filePaths)

    outLogPathDir = os.path.join( os.path.dirname(__file__), '../output-Kinetics/' + inputVideoDir.split("/")[-2] + '/')
    if not os.path.exists(outLogPathDir):
        os.makedirs(outLogPathDir)
    outLogPath = outLogPathDir + inputVideoDir.split("/")[-2] + '_test_log.csv'
    
    logLevel = logging.WARNING 
    logger = loggingSetting(outLogPath, logLevel)
    
    for fps in frameRates:
        for reso in resolutions:
            
            for fpath in filePaths:
                fileNameOut = " ".join(fpath.split("/")[-1].split(".")[:-1]) + "_basketballDunk_out.mp4"
                print ("fileName: ", fileNameOut)
                
                #actionDunkCnt, elapsedTime, NUMFRAMES = detectBasketballDunk(fpath, fileNameOut, fps, reso)
                actionDunkCnt, elapsedTime, NUMFRAMES = detectBasketballDunkKFrameFixedWindow(fpath, fileNameOut, fps, reso, K, ratioKFrame)

                
                actionTF = False           # TrueFalse
                if actionDunkCnt >= 1:
                    actionTF = True
                logger.warning('filename: %s, NumFrame: %s, frame rate: %s, resolution: %s, elapsedTotalTime: %s, timePerFrame:%s, actionResult: %s, ActionTrueFalse: %s ',
                               fpath.split("/")[-1], NUMFRAMES, fps, reso[1], elapsedTime, elapsedTime/NUMFRAMES, actionDunkCnt, actionTF)


def readConfigurationResult(inputFile):
    '''
    return
    fpsLst, resolutionLst, sPFTimeLst, accuracyEachConfigLst
    e.g.
     ['30', '30', '30', '30', '30', '10', '10', '10', '10', '10', '5', '5', '5', '5', '5', '2', '2', '2', '2', '2', '1', '1', '1', '1', '1'] 
     ['720', '600', '480', '360', '240', '720', '600', '480', '360', '240', '720', '600', '480', '360', '240', '720', '600', '480', '360', '240', '720', '600', '480', '360', '240'] 
     [0.244, 0.126, 0.053, 0.015, 0.007, 0.24, 0.126, 0.053, 0.015, 0.007, 0.243, 0.127, 0.052, 0.015, 0.007, 0.241, 0.126, 0.053, 0.015, 0.007, 0.242, 0.126, 0.053, 0.015, 0.007]
     [0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.1, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.1, 0.0, 0.0, 0.5, 0.5, 0.1, 0.0, 0.0]

    '''

    fpsLst = []                 # frame rate
    resolutionLst = []          # image size
    sPFTimeLst = []             # second per frame
    
    # get accuracy
    actionDetected = 0
    accuracyEachConfigLst = []
    
    numVideos = 10
    cnt = 0
    with open(inputFile, "rt", encoding='ascii') as inputFile:
        rows = csv.reader(inputFile)
        
        averageSPF = 0.0
        sumSPF = 0.0
        for row in rows :
            #print (row)
            cnt += 1
            if cnt == 1:
                #numFrame = row[2].split(':')[1]
                frameRate = row[3].split(':')[1].strip()
                resoPixels = row[4].split(':')[1].strip()
                #totalTime = row[5].split(':')[1]
                fpsLst.append(frameRate)
                resolutionLst.append(resoPixels)
                
            secPerFrame = float(row[6].split(':')[1].strip())
            #actionDunkCnt = row[7].split(':')[1]    
            sumSPF += secPerFrame
            
            ActionTrueFalse = row[8].split(':')[1].strip()
            
            if ActionTrueFalse == "True":
                actionDetected += 1
                
                
            if cnt == numVideos:
                averageSPF = round(sumSPF/numVideos, 3)
                sPFTimeLst.append(averageSPF)
                sumSPF = 0.0
                
                accuracy = actionDetected/numVideos
                accuracyEachConfigLst.append(accuracy) 
                actionDetected = 0
                
                cnt = 0

    print ("averageSPF: ",  fpsLst, resolutionLst, sPFTimeLst, accuracyEachConfigLst)
            
    return fpsLst, resolutionLst, sPFTimeLst, accuracyEachConfigLst


def plotConfigImpact():
    '''
    plot 1:
    y axis:  accuracy 
    x axis: delay time; SPF second per frame  
    value:  frame rate fps;  different frame's accuracy and SPF value pair
    
    only plot maximum resolution 720p
    
    plot 2:
    y axis:  accuracy 
    x axis: delay time; SPF second per frame  
    value:  image size: resolution;  different resolution's accuracy and SPF value pair

    only plot maximum frame rate = 30fps 
    '''
    
    inputFile = '/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/output-Kinetics/testVideo01_trimmed_10videos/testVideo01_trimmed_10videos_test_log.csv'
    
    fpsLst, resolutionLst, sPFTimeLst, accuracyEachConfigLst = readConfigurationResult(inputFile)
    
    knobNum = 5    # len(frameRates)
    #plot 1 second per frame vs accuracy;   fixed resolution = 720p
    xLst = []           # SPF
    yLst = []           # SPF
    i = 0         # resolution, framerate start index
    for i in range(i, len(fpsLst), knobNum):
        print ("fpsLst: ", fpsLst[i])
        xLst.append(sPFTimeLst[i])
        yLst.append(accuracyEachConfigLst[i])
        
    print ("xLst: ",  xLst, yLst)
    
    plt.plot(xLst, yLst, 'o-')
    plt.title('Impact of frame rate')
    plt.xlabel('Processing speed--Second per frame ')
    plt.ylabel('Accuracy')
    
              
if __name__== "__main__":
    exec(sys.argv[1])
