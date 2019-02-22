#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:00:56 2018

@author: fubao
"""

# test the effecct of tradtional opencv on basketball dunk action
# in a systematical way

#from dunKActionDetectionObjectDectionCV import detectBasketballDunk
from dunKActionDetectionObjectDectionCV import detectBasketballDunkKFrameFixedWindow

from profilingCommon import ProfileVideoCls

import logging
import glob
import sys
import os 
import csv

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dataComm import loggingSetting


def testAllVideosDir(inputVideoDir):
        
    K = 6     # each K frames
    ratioKFrame = 0.1    # how many frames detected as dunk over K frame
    frameRates = [25, 10, 5, 2, 1]   # [5]   #[25, 10, 5, 2, 1]            #  [25]    #  [25, 10, 5, 2, 1]    # [30],  [30, 10, 5, 2, 1] 
    resoPixels = [720, 600, 480, 360, 240]   # [240] # [720, 600, 480, 360, 240]    # [240]       # [720, 600, 480, 360, 240]    #   [720]     # [720, 600, 480, 360, 240]  #  [720]    # [720, 600, 480, 360, 240]            #  16: 9
    resolutions = [(w*16//9, w) for w in resoPixels]
    #inputVideoDir = "../inputData/kinetics600/videos-dunkBasketball/testVideo01_trimmed_10videos/"
    #inputVideoDir = "../inputData/kinetics600/videos-dunkBasketball/testVideo01_trimmed_30videos/"
    #inputVideoDir = "../inputData/kinetics600/videos-dunkBasketball/testVideo01_trimmed_50videos/"
    
    # get video from inputVideoDir
    filePaths = glob.glob(inputVideoDir + "*.mp4")
    #print ("files: ", filePaths)

    outLogPathDir = os.path.join( os.path.dirname(__file__), '../output-Kinetics/' + inputVideoDir.split("/")[-2] + '/')
    if not os.path.exists(outLogPathDir):
        os.makedirs(outLogPathDir)
    outLogPath = outLogPathDir + inputVideoDir.split("/")[-2] + '_CCV_test_log.csv'
    
    
    
    logLevel = logging.WARNING 
    logger = loggingSetting(outLogPath, logLevel)
    
    for fps in frameRates:
        for reso in resolutions:
            for fpath in filePaths:
                fileNameOut = " ".join(fpath.split("/")[-1].split(".")[:-1]) + "_basketballDunk_out.mp4"
                print ("fileName: ", fileNameOut)
                
                #actionDunkCnt, elapsedTime, NUMFRAMES = detectBasketballDunk(fpath, fileNameOut, fps, reso)
                actionDunkCnt, elapsedTime, realFrameNum, NUMFRAMESTOTAL = detectBasketballDunkKFrameFixedWindow(fpath, fileNameOut, fps, reso, K, ratioKFrame)

                
                actionTF = False           # TrueFalse
                if actionDunkCnt >= 1:
                    actionTF = True
                logger.warning('filename: %s, NumFrames: %s, frame rate: %s, resolution: %s, elapsedTotalTime: %s, timePerFrame:%s, actionResult: %s, ActionTrueFalse: %s ',
                               fpath.split("/")[-1], realFrameNum, fps, reso[1], elapsedTime, elapsedTime/NUMFRAMESTOTAL, actionDunkCnt, actionTF)


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

    numVideos = 50           # 30   10
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
                

    print ("fps, resolution, spf, acc: ",  fpsLst, resolutionLst, sPFTimeLst, accuracyEachConfigLst)
            
    return fpsLst, resolutionLst, sPFTimeLst, accuracyEachConfigLst


def readConfigurationResultIntoSequence(inputFile):
    '''
    read profile file result into ProfileVideoCls data structure
    
    return a list of ProfileVideoCls
    
    
    '''

    profileResultLst = []       #list of ProfileVideoCls

    #fpsLst = []                 # frame rate
    #resolutionLst = []          # image size
    #sPFTimeLst = []             # second per frame
    
    # get accuracy
    actionDetected = 0
    #accuracyEachConfigLst = []
    
        
    #  ['cameraNo', 'frameStartNo', 'resolution', 'frameRate', 'accuracy', 'costSPF']

    modelMethod = "CCV"    # classical computer vision now
    
    cameraNo = 1           # currently only 1 to test first
    frameStartNo = 0
    
    numVideos = 50           # 30   10
    cnt = 0
    with open(inputFile, "rt", encoding='ascii') as inputFile:
        rows = csv.reader(inputFile)
        
        averageSPF = 0.0
        sumSPF = 0.0
        
        proClsObj = ProfileVideoCls()         # profile result class

        for row in rows :
            #print (row)
            cnt += 1
            if cnt == 1:
                #numFrame = row[2].split(':')[1]
                frameRate = row[3].split(':')[1].strip()
                resoPixels = row[4].split(':')[1].strip()
                #totalTime = row[5].split(':')[1]
                #fpsLst.append(frameRate)
                #resolutionLst.append(resoPixels)
                
                proClsObj.cameraNo = cameraNo
                proClsObj.frameStartNo = frameStartNo
                proClsObj.frameRate = frameRate
                proClsObj.resolution = resoPixels
                proClsObj.modelMethod = modelMethod
                
            secPerFrame = float(row[6].split(':')[1].strip())
            #actionDunkCnt = row[7].split(':')[1]    
            sumSPF += secPerFrame
            
            ActionTrueFalse = row[8].split(':')[1].strip()
            
            if ActionTrueFalse == "True":
                actionDetected += 1
                
                
            if cnt == numVideos:
                averageSPF = round(sumSPF/numVideos, 3)
                #sPFTimeLst.append(averageSPF)
                proClsObj.costSPF = averageSPF
                
                accuracy = actionDetected/numVideos
                #accuracyEachConfigLst.append(accuracy) 
                proClsObj.accuracy = accuracy
                
                profileResultLst.append(proClsObj)
                
                proClsObj = ProfileVideoCls()         # reinitialize
                
                sumSPF = 0.0  
                actionDetected = 0  # for the next video clips
                cnt = 0
                
                frameStartNo += 1

            
    return profileResultLst



def readMultipleConfigurationResultIntoSequence(inputFile1, inputFile2):
    '''
    read profile file1 from CCV , file2 from DNN result into ProfileVideoCls data structure
    
    return a list of ProfileVideoCls
    
    
    '''

    profileResultLst = []       #list of ProfileVideoCls

    #inputFileLst = [inputFile1, inputFile2]
    
    fileInd = 0
    
    while(fileInd < 2):           # 2 files
        
        
        # get accuracy
        actionDetected = 0
        #accuracyEachConfigLst = []
        
        #  ['cameraNo', 'frameStartNo', 'resolution', 'frameRate', 'accuracy', 'costSPF']
    
        if fileInd == 0:
            modelMethod = "CCV"    # classical computer vision now
            inputFile = inputFile1
        else:
            modelMethod = "DNN"    # classical computer vision now
            inputFile = inputFile2
            
        cameraNo = 1           # currently only 1 to test first
        frameStartNo = 0
        
        numVideos = 50               # 30   10
        cnt = 0
        
        
        with open(inputFile, "rt", encoding='ascii') as inputFile:
            rows = csv.reader(inputFile)
            
            averageSPF = 0.0
            sumSPF = 0.0
            
            proClsObj = ProfileVideoCls()         # profile result class
    
            for row in rows :
                #print (row)
                cnt += 1
                if cnt == 1:
                    #numFrame = row[2].split(':')[1]
                    frameRate = row[3].split(':')[1].strip()
                    resoPixels = row[4].split(':')[1].strip()
                    #totalTime = row[5].split(':')[1]
                    #fpsLst.append(frameRate)
                    #resolutionLst.append(resoPixels)
                    
                    proClsObj.cameraNo = cameraNo
                    proClsObj.frameStartNo = frameStartNo
                    proClsObj.frameRate = frameRate
                    proClsObj.resolution = resoPixels
                    proClsObj.modelMethod = modelMethod
                        
                secPerFrame = float(row[6].split(':')[1].strip())
                #actionDunkCnt = row[7].split(':')[1]    
                sumSPF += secPerFrame
                
                ActionTrueFalse = row[8].split(':')[1].strip()
                
                if ActionTrueFalse == "True":
                    actionDetected += 1
                    
                    
                if cnt == numVideos:
                    averageSPF = round(sumSPF/numVideos, 3)
                    #sPFTimeLst.append(averageSPF)
                    proClsObj.costSPF = averageSPF
                    
                    accuracy = actionDetected/numVideos
                    #accuracyEachConfigLst.append(accuracy) 
                    proClsObj.accuracy = accuracy
                    
                    profileResultLst.append(proClsObj)
                    
                    proClsObj = ProfileVideoCls()         # reinitialize
                    
                    sumSPF = 0.0  
                    actionDetected = 0  # for the next video clips
                    cnt = 0
                    
                    frameStartNo += 1
                    
        fileInd += 1
    return profileResultLst


def plotConfigImpact(inputFile):
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

    only plot maximum frame rate = 25fps 
    '''
    
    #inputFile = '/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/output-Kinetics/testVideo01_trimmed_10videos/testVideo01_trimmed_10videos_test_log.csv'
    #inputFile = '/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/output-Kinetics/testVideo01_trimmed_30videos/testVideo01_trimmed_30videos_test_log.csv'

    fpsLst, resolutionLst, sPFTimeLst, accuracyEachConfigLst = readConfigurationResult(inputFile)
    
    knobNum = 5  # 5    # len(frameRates)
    
    #plot 1 second per frame vs accuracy;   fixed resolution = 720p
    
    
    xSPFAllResoLsts = []           # All different FPS for one resolution
    yAccAllResoLsts = []           # Accu
    
    resNum = 0
    
    while (resNum < len(resolutionLst)):
        start = resNum         # resolution of 720  framerate start indexd
        xSPFLst = []
        yAccLst = []
        for i in range(start, len(fpsLst), knobNum):
            #print ("fpsLst: ", fpsLst[i])
            xSPFLst.append(sPFTimeLst[i])
            yAccLst.append(accuracyEachConfigLst[i])
        
        xSPFAllResoLsts.append(xSPFLst)
        yAccAllResoLsts.append(yAccLst)
        resNum += 1
    print ("resolution 720: ",  xSPFAllResoLsts[0], yAccAllResoLsts[0])
    
    
    '''
    ax = plt.subplot() # Defines ax variable by creating an empty plot
    # Set the tick labels font
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #label.set_fontname('Arial')
        label.set_fontsize(15)
    '''
    
    

    plt.figure(1)
    
    for i in range(0, len(xSPFAllResoLsts)):            # len(xSPFAllResoLsts)
        plt.plot(xSPFAllResoLsts[i][::-1], yAccAllResoLsts[i][::-1], 'o')         #'o-'
        
    plt.title('Impact of frame rates: ' + '[1, 2, 5, 10, 25]', size=16)
    plt.xlabel('CPU cost--Second Per Frame ', size=16)
    plt.ylabel('Accuracy', size=16)
    plt.savefig("/".join(inputFile.split("/")[:-1]) + "/" + "CCV_impactFrameRate1.pdf")


    
    
    xResoAllResoLsts = []           # reso
    yAccAllResoLsts = []           # Accu
    
    resNum = 0
    while (resNum < len(resolutionLst)):
        start = resNum              # fps of 25  framerate start indexd
        xResoLst = []
        yAccLst = []
        for i in range(start, knobNum+start, 1):
            #print ("fpsLst: ", fpsLst[i])
            xResoLst.append(sPFTimeLst[i])
            yAccLst.append(accuracyEachConfigLst[i])
        
        xResoAllResoLsts.append(xResoLst)
        yAccAllResoLsts.append(yAccLst)
        resNum += knobNum       # add knobNum= 5
    print ("frame rate 25: ",  xResoAllResoLsts[0], yAccAllResoLsts[0])
    

        
    plt.figure(2)
    
    for i in range(0, len(xResoAllResoLsts)):            # len(xResoAllResoLsts)
        plt.plot(xResoAllResoLsts[i][::-1], yAccAllResoLsts[i][::-1], 'o')
    plt.title('Impact of resolutions: ' + '[240, 360, 480, 600, 720]', size=16)      # [240, 360, 480, 600, 720]
    plt.xlabel('CPU cost--Second Per Frame ', size=16)
    plt.ylabel('Accuracy', size=16)
    plt.savefig("/".join(inputFile.split("/")[:-1]) + "/" + "CCV_impactReso1.pdf")
    
    #plt.show()

def test1(inputFile1, inputFile2):
     # test
    #inputFile1 = '../output-Kinetics/testVideo01_trimmed_50videos/testVideo01_trimmed_50videos_test_log.csv'
    #profileResultLst = readConfigurationResultIntoSequence(inputFile1)
    
    profileResultLst = readMultipleConfigurationResultIntoSequence(inputFile1, inputFile2)

    for i, proObj in enumerate(profileResultLst):
        print ("prob res: ", proObj.cameraNo, proObj.frameStartNo, proObj.resolution, 
               proObj.frameRate, proObj.modelMethod, proObj.accuracy, proObj.costSPF)
        
    #  ['cameraNo', 'frameStartNo', 'resolution', 'frameRate', 'accuracy', 'costSPF']

    '''
    # test to sort by an element in  costSPF
    profileResLstSortAcc = sorted(profileResultLst, key=lambda profCls: profCls.accuracy, reverse=True)
    for i, proObj in enumerate(profileResLstSortAcc):
        print ("prob res sorted: ", proObj.cameraNo, proObj.frameStartNo, proObj.resolution, 
               proObj.frameRate, proObj.modelMethod, proObj.accuracy, proObj.costSPF)
    
    '''
    
if __name__== "__main__":
    exec(sys.argv[1])
    
   
    
        