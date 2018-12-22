#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:00:56 2018

@author: fubao
"""

# test the effecct of tradtional opencv on basketball dunk action
# in a systematical way
from dunKActionDetectionObjectDectionCV import detectBasketballDunk

import glob


def testAllVideosDir():
        
    frameRates =  [30]     # [1,2,5,10,30]
    resolutions = [720]    # [720, 600, 480, 360, 240]            #  16: 9
    inputVideoDir = "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/openCVMethod/inputData/kinetics600/videos-dunkBasketball/testVideo01_10videos/"
    # get video from inputVideoDir
    filePaths = glob.glob(inputVideoDir + "*.mp4")
    #print ("files: ", filePaths)
    
    resolutionPx = 0
    
    for fps in frameRates:
        for res in resolutions:
            
            for fpath in filePaths:
                fileNameOut = " ".join(fpath.split("/")[-1].split(".")[:-1]) + "_basketballDunk_out.mp4"
                print ("fileName: ", fileNameOut)
                
                detectBasketballDunk(fpath, fileNameOut, fps, res)