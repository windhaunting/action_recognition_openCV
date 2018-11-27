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

def draw_circles(storage, output):
    circles = np.asarray(storage)
    for circle in circles:
        Radius, x, y = int(circle[0][3]), int(circle[0][0]), int(circle[0][4])
        cv2.Circle(output, (x, y), 1, cv2.RGB(0, 255, 0), -1, 8, 0)
        cv2.Circle(output, (x, y), Radius, cv2.RGB(255, 0, 0), 3, 8, 0)


def testImage(inputImage, outputImage):
    
    frame = cv2.imread(inputImage)
    #print ("testImage ", frame)
    
    # first detect human
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


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
            
            
    # convert to HSV space
    im_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # take only the orange, highly saturated, and bright parts
    im_hsv = cv2.inRange(im_hsv, (7,50,50), (11,1,255)) # (0,0,0), (255,255,255))   #(7,150,150), (11,255,255))
    cv2.imshow("im_hsv", im_hsv)
    cv2.imwrite("im_hsv.jpg", im_hsv)
    
    
    #detect circle
    out = frame.copy()
    im_orange = frame.copy()
    im_orange[im_hsv==0] = 0
    
    gray = cv2.cvtColor(im_orange, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray,(5,5),0);
    gray = cv2.medianBlur(gray,5)
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,3.5)
    kernel = np.ones((3, 3),np.uint8)
    gray = cv2.erode(gray,kernel,iterations = 1)
    gray = cv2.dilate(gray,kernel,iterations = 1)
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 28, 3500, param1=15, param2=85, minRadius=1, maxRadius=60)


    # To show the detected orange parts:
    im_orange = frame.copy()
    im_orange[im_hsv==0] = 0
    # cv2.imshow('im_orange',im_orange)
    
    # Perform opening to remove smaller elements
    element = np.ones((5,5)).astype(np.uint8)
    im_hsv = cv2.erode(im_hsv, element)
    im_hsv = cv2.dilate(im_hsv, element)
    
    #points = np.dstack(np.where(im_hsv>0)).astype(np.float32)
    # fit a bounding circle to the orange points
    #center, radius = cv2.minEnclosingCircle(points)
    
    print ("circles: ", circles)
    
    if circles is not None:
        	# convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
         
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(gray, (x, y), r, (0, 0, 255), 4)
            cv2.putText(gray, "circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 6)  # Text in black

            #cv2.rectangle(out, (x - 25, y - 25), (x + 25, y + 25), (255, 0, 0), -1)
            cv2.circle(out, (x, y), r, (0, 0, 255), 4)
            
        outImg = np.vstack([out, im_orange])
        cv2.imwrite(outputImage, outImg)

    # draw this circle
    #cv2.circle(frame, (int(center[1]), int(center[0])), int(radius), (255,0,0), thickness=3)
    
    #out = np.vstack([im_orange,frame])

    #cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    #cv2.imwrite('maks', mask)
    #cv2.imwrite(outputImage, out)
    
    
    '''
    # detect circles in the image
    output = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
     
    print ("circles: ", circles, len(circles))
    # ensure at least some circles were found
    if circles is not None:
        	# convert the (x, y) coordinates and radius of the circles to integers
        	circles = np.round(circles[0, :]).astype("int")
         
        	# loop over the (x, y) coordinates and radius of the circles
        	for (x, y, r) in circles:
        		# draw the circle in the output image, then draw a rectangle
        		# corresponding to the center of the circle
        		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255,0,0), -1)
         
        	# show the output image
        	cv2.imshow("output", np.hstack([frame, output]))
        	cv2.waitKey(0)

    '''
    
    '''
   
    storage = cv.CreateMat(orig.width, 1, cv.CV_32FC3)
    #use canny, as HoughCircles seems to prefer ring like circles to filled ones.
    cv.Canny(processed, processed, 5, 70, 3)
    #smooth to reduce noise a bit more
    cv.Smooth(processed, processed, cv.CV_GAUSSIAN, 7, 7)
    
    cv.HoughCircles(processed, storage, cv.CV_HOUGH_GRADIENT, 2, 32.0, 30, 550)
    draw_circles(storage, orig)
    
    cv.imwrite('found_basketball.jpg',orig)
    '''
 
    
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
 
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    startFrame = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print ("no frame exit here 1, total frames ", ret)
            break
        
        startFrame += 1
        cv2.imwrite('basketDunkImgInput' + str(startFrame) + '.jpg', frame)
        
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
        # detect basketball and track it
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
        '''
        #print("original frame: ", type(frame), frame.shape)
        #cv2.imshow('original image',frame)
        frame = fgbg.apply(frame)
        
        #print("foreground frame: ", type(frame), frame.shape)
        #im_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # take only the orange, highly saturated, and bright parts
        im_hsv = cv2.inRange(frame,100, 130)   # (0,0,155), (255,255,255))           # (7,180,180), (11,255,255))
        
        # To show the detected orange parts:
        im_orange = frame.copy()
        im_orange[im_hsv==0] = 0
        cv2.imshow('im_orange',im_orange)
        
        # Perform opening to remove smaller elements
        element = np.ones((5,5)).astype(np.uint8)
        im_hsv = cv2.erode(im_hsv, element)
        im_hsv = cv2.dilate(im_hsv, element)
        
        points = np.dstack(np.where(im_hsv>0)).astype(np.float32)
        
        # fit a bounding circle to the orange points
        center, radius = cv2.minEnclosingCircle(points)
        # draw this circle
        cv2.circle(frame, (int(center[1]), int(center[0])), int(radius), (255,0,0), thickness=3)
        
        out = np.vstack([im_orange,frame])
        cv2.imwrite('out.png',out)
        
        # Display the resulting frame
        cv2.imshow('Video out', out)
        #outVideo.write(out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    outVideo.release()



if __name__== "__main__":
    exec(sys.argv[1])
    