#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:06:22 2018

@author: fubao
"""

# use  HOG + SVM to do detect human


# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
import sys
import os
import imutils
from collections import deque


from dataComm import loggingSetting
from dataComm import readVideo



def detectHuman(videoPath, outputVideoName):
    '''
    HOG SVM Pretrained
    videoPath: input a vdideo
    
    '''
    print ("videoPath, model path: ", videoPath)
    
    cap = readVideo(videoPath)
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    NUMFRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    print ('cam stat: %s, %s, %s, %s ', fps, WIDTH, HEIGHT, NUMFRAMES)
    
    # outputVideoName =  "UCF101_v_longJump_g01_c01_out.avi"   # "UCF101_v_BasketballDunk_g01_c01_out.avi"  # "humanRunning_JHMDB_output_001.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')   # MJPG
    outputVideoDir = os.path.join( os.path.dirname(__file__), '../output/')
    outVideo = cv2.VideoWriter(outputVideoDir +"HOG_" + outputVideoName, fourcc, 5,  (int(WIDTH), int(HEIGHT)))
    
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    basketballLower = (7,180,180)
    basketballUpper = (11,255,255)
    pts = deque(maxlen=1000)                # buffer size
 

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print ("no frame exit here 1, total frames ", ret)
            break
        
        #orig = frame.copy()

        #imScale = cv2.resize(frame,(320,240)) # Downscale to improve frame rate
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4),
        		padding=(8, 8), scale=1.05)
         
        # draw the original bounding boxes
        #for (x, y, w, h) in rects:
        #	 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
         
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
         
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
         
        
        '''
        detect basketball and track it
        '''
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
         
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, basketballLower, basketballUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
    
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        im, cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        		cv2.CHAIN_APPROX_SIMPLE)
        if cnts is None:
            print ("cnts: ", cnts)

            continue
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        center = None
         
        
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
        	   # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
         
            # only proceed if the radius meets a minimum size
            if radius > 10:
        			# draw the circle and centroid on the frame,
        			# then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
        				(0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
 
        # update the points queue
        pts.appendleft(center)
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
    exec(sys.argv[1])
    