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

from dataComm import loggingSetting


def testAllVideosDir():
        
    K = 6     # each K frames
    ratioKFrame = 0.3    # how many frames detected as dunk over K frame
    frameRates = [30]  #  [30, 10, 5, 2, 1]    # [30],  [30, 10, 5, 2, 1] 
    resoPixels = [720]     # [720, 600, 480, 360, 240]  #  [720]    # [720, 600, 480, 360, 240]            #  16: 9
    resolutions = [(w*16//9, w) for w in resoPixels]
    inputVideoDir = "../inputData/kinetics600/videos-dunkBasketball/testVideo01_trimmed_10videos/"
    # get video from inputVideoDir
    filePaths = glob.glob(inputVideoDir + "*.mp4")
    #print ("files: ", filePaths)

    outLogPath = os.path.join( os.path.dirname(__file__), '../output-Kinetics/' + inputVideoDir.split("/")[-2] + '_test_log.csv')
    
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
                logger.warning('filename: %s, NumFrame: %s, frame rate: %s, resolution: %s, elapsedTotalTime: %s, timePerFrame:%s,  actionResult: %s, ActionTrueFalse: %s ',
                               fpath.split("/")[-1], NUMFRAMES, fps, reso, elapsedTime, elapsedTime/NUMFRAMES, actionDunkCnt, actionTF)
                
if __name__== "__main__":
    exec(sys.argv[1])
