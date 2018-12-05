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
import math

from collections import deque

from dataComm import loggingSetting
from dataComm import readVideo

def draw_circles(storage, output):
    circles = np.asarray(storage)
    for circle in circles:
        Radius, x, y = int(circle[0][3]), int(circle[0][0]), int(circle[0][4])
        cv2.Circle(output, (x, y), 1, cv2.RGB(0, 255, 0), -1, 8, 0)
        cv2.Circle(output, (x, y), Radius, cv2.RGB(255, 0, 0), 3, 8, 0)


    
def findContour(inputImage, outputImage):
    
    '''
    1st detect human
    2nd extend the detection region in a threshold (20%) larger
    '''
    frame = cv2.imread(inputImage)

    cv2.imshow('Original Image', frame)
    #cv2.waitKey(0)
    
    bilateral_filtered_image = cv2.bilateralFilter(frame, 5, 175, 175)
    cv2.imshow('Bilateral', bilateral_filtered_image)
    #cv2.waitKey(0)
    
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    cv2.imshow('Edge', edge_detected_image)
    #cv2.waitKey(0)
    
    contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # for basket  .03** cv2.arcLength(contour, False)  ((len(approx) >=10) & (area > 100) ):
    # for basketball 
    contour_list = []
    for contour in contours:
        #perfect circle area
        perfectCircleArea = math.pi* (cv2.arcLength(contour, False)//2)**2
        
        epsilon = .1 * cv2.arcLength(contour, False)
        approx = cv2.approxPolyDP(contour,epsilon, False)
        area = cv2.contourArea(contour)
        
        #if area == 0:
        #    continue
        #if abs(perfectCircleArea-area)< 0.9:
        #    contour_list.append(contour)
        #print ("area: ", len(approx))
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)
        #cv2.circle(frame,center,radius,(0,255,0),2)
    
        if ((len(approx) ==12) & (area > 150) ):
            contour_list.append(contour)
            print ("perfectCircleArea: ", approx, len(approx), area)
           
        
    cv2.drawContours(frame, contour_list,  -1, (255,0,0), 2)
    cv2.imshow('Objects Detected',frame)
    cv2.waitKey(0)
    
    
def testImage(inputImage, outputImage):
    '''
    test a single image
    detect edge, color range, 
    '''
    
    frame = cv2.imread(inputImage)
    height, width, channels = frame.shape
    print (height, width, channels)
    
    #image_resize = cv2.resize(frame, (640, 360)) 


    #print ("testImage ", frame)
    cv2.imshow('Original Image', frame)

    # first detect human
    hog = cv2.HOGDescriptor()
    #hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(gray, winStride=(3, 3),
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
    
    
        crop_img = frame[yA:yB, xA:xB]          # cropped image only containing detected human
        cv2.imshow("cropped", crop_img)
    
    cv2.imshow('detected human image', frame)

    
    # convert to HSV space
    im_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # take only the orange, highly saturated, and bright parts
    im_hsv = cv2.inRange(im_hsv, (7,50,50), (255,255,255)) # (0,0,0), (255,255,255))   #(7,150,150), (11,255,255))
    cv2.imshow("im_hsv", im_hsv)
    cv2.imwrite(outputImage + "_im_hsv.jpg", im_hsv)

    print ("extracted color range ")

    #detect circle
    out = frame.copy()
    im_orange = frame.copy()
    im_orange[im_hsv==0] = 0
    
    #gray = cv2.cvtColor(im_orange, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(im_orange, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray,(5,5),0);
    gray = cv2.medianBlur(gray,5)
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,3.5)
    kernel = np.ones((3, 3),np.uint8)
    gray = cv2.erode(gray,kernel,iterations = 1)
    gray = cv2.dilate(gray,kernel,iterations = 1)
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 11, 600, param1=15, param2=85, minRadius=1, maxRadius=120)


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
            cv2.circle(out, (x, y), r, (0, 0, 255), 4)
            cv2.putText(out, "circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 6)  # Text in black

            #cv2.rectangle(out, (x - 25, y - 25), (x + 25, y + 25), (255, 0, 0), -1)
            #cv2.circle(out, (x, y), r, (0, 0, 255), 4)
            
        outImg = np.vstack([out, im_orange])
        cv2.imwrite(outputImage, outImg)


    cv2.waitKey(0)


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


def judgeBasketDunkAction(humanCenter, ballCenter, basketCenter, thresholdX):
    '''
    according to these postion to decide a frame has dunk or not 
    each position use (x1, y1, x2, y2) left upper and right coordinate
    '''
    
    actionDunk = False
    humanCenterX = humanCenter[0]
    humanCenterY = humanCenter[1]
    
    ballCenterX = ballCenter[0]
    ballCenterY = ballCenter[1]

    basketCenterX = basketCenter[0]
    basketCenterY = basketCenter[1]
    
    if (ballCenterY >= basketCenterY)    \
        and (basketCenterY >= humanCenterY)   \
        and (abs(basketCenterX - humanCenterX) < thresholdX):
        actionDunk  = True

    return actionDunk



def detectBasket(frame, xA, yA, xB, yB, startFrame,  outDir):
    '''
    detect the basket
    "xA, yA, xB, yB " is a rectangle of human detected
    '''
    
    rectImg = frame[yA:yB, xA:xB]

    bilateral_filtered_image = cv2.bilateralFilter(rectImg, 5, 175, 175)
    #cv2.imshow('Bilateral', bilateral_filtered_image)
    #cv2.waitKey(0)
    
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    #cv2.imshow('Edge', edge_detected_image)
    #cv2.waitKey(0)
    
    contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # for basket  .03** cv2.arcLength(contour, False)  ((len(approx) >=10) & (area > 100) ):
    # for basket
    basketCenterLst = []
    contour_list = []
    for contour in contours:
        #perfect circle area        
        epsilon = .05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour,epsilon, True)
        area = cv2.contourArea(contour)
        
        #if area == 0:
        #    continue
        #if abs(perfectCircleArea-area)< 0.9:
        #    contour_list.append(contour)
        #print ("area: ", len(approx), area)

        if ((len(approx) >= 6) and (area > 13) and (area < 16)):
            contour_list.append(contour)
            #print ("perfectEllipseArea: ", len(approx), area)
            ellipse= cv2.fitEllipse(contour)
            (center,axes,orientation) = ellipse
            #majoraxis_length = max(axes)
            #minoraxis_length = min(axes)
            #eccentricity=(np.sqrt(1-(minoraxis_length/majoraxis_length)**2))
            cv2.ellipse(rectImg, ellipse, (0,255,255),2)
            cv2.putText(rectImg, "E", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255), 6)  # Text in black
            basketCenterLst.append((int(center[0]) + xA, int(center[1]) + yA))

    if len(contour_list) != 0:
        print ("BASKET detected frame ", startFrame)

        #cv2.drawContours(rectImg, contour_list,  -1, (255,0,0), 2)
        frameOutFile = outDir + 'B2_basket_' + str(startFrame) + '_' + str(xA) + '-' + str(yA) + '.jpg'
        if not os.path.exists(frameOutFile):
            cv2.imwrite(frameOutFile, rectImg)
    else:
        t = 1
        #print ("No basket detected frame ", startFrame)

    return basketCenterLst
    

def detectBall(frame, xA, yA, xB, yB, startFrame,  outDir):
    '''
    detect basketball
    "xA, yA, xB, yB " is a rectangle of human detected
    '''
    
    rectImg = frame[yA:yB, xA:xB]

    bilateral_filtered_image = cv2.bilateralFilter(rectImg, 5, 175, 175)
    #cv2.imshow('Bilateral', bilateral_filtered_image)
    #cv2.waitKey(0)
    
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    #cv2.imshow('Edge', edge_detected_image)
    #cv2.waitKey(0)
    
    contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # for basket  .03** cv2.arcLength(contour, False)  ((len(approx) >=10) & (area > 100) ):
    # for basketball 
    
    ballCenterLst = []
    contour_list = []
    for contour in contours:
        #perfect circle area        
        epsilon = 0.1 * cv2.arcLength(contour, False)
        approx = cv2.approxPolyDP(contour,epsilon, False)
        area = cv2.contourArea(contour)
        
        #if area == 0:
        #    continue
        #if abs(perfectCircleArea-area)< 0.9:
        #    contour_list.append(contour)
        #print ("area: ", len(approx))
       
        if ((len(approx) >= 5) and (area > 6) and (area < 20)):
            contour_list.append(contour)
            #print ("perfectCircleArea: ", len(approx), area)
            (x,y),radius = cv2.minEnclosingCircle(contour)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(rectImg, center,radius,(0,165,255),2)
            cv2.putText(rectImg, "c" + str(round(area, 1)), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,165,255), 6)  # Text in black
            ballCenterLst.append((int(center[0]) + xA, int(center[1]) + yA))
            
    if len(contour_list) != 0:
        print ("BASKETBALL detected frame ", startFrame)

        #cv2.drawContours(rectImg, contour_list,  -1, (255,0,0), 2)
        frameOutFile = outDir + 'B1_basketeBall_' + str(startFrame) + '_' + str(xA) + '-' + str(yA) + '.jpg'
        if not os.path.exists(frameOutFile):
            cv2.imwrite(frameOutFile, rectImg)
    else:
        t = 1
        #print ("No basketball detected frame ", startFrame)

    
    return ballCenterLst
    
    
def detectBasketDunk(videoPath, outputVideoName):
    '''
    detect basketball dunk
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
    finalOutDir = outputVideoDir + outputVideoName + "/"
    if not os.path.exists(finalOutDir):
        os.makedirs(finalOutDir)
    
    outVideo = cv2.VideoWriter(finalOutDir +"HOG_" + outputVideoName, fourcc, 5,  (int(WIDTH), int(HEIGHT)))
    
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    startFrame = 0
    
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print ("no frame exit here 1, total frames ", ret)
            break
        
        startFrame += 1
        
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
        picks = non_max_suppression(rects, probs=None, overlapThresh=0.65)
         
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in picks:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
        frameOutFile = finalOutDir + 'basketDunkImg' + str(startFrame) + '.jpg'
        if not os.path.exists(frameOutFile):
            cv2.imwrite(frameOutFile, frame)


        # detect basketball around extended human region
        for (xA, yA, xB, yB) in picks:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            
            humanCenter = ((xA +xB)/2, (yA+yB)/2)
            #crop image
            ballCenterLst = detectBall(frame, int(0.8*xA), int(0.5*yA), int(1.2*xB), int(0.7*yB), startFrame,  finalOutDir)
            basketCenterLst = detectBasket(frame, int(0.3*xA), int(0.1*yA), int(2*xB), int(0.5*yB), startFrame,  finalOutDir)
            
            for ballCenter in ballCenterLst:
                for basketCenter in basketCenterLst:
                    actionDunk = judgeBasketDunkAction(humanCenter, ballCenter, basketCenter, humanCenter[0])
                    
                    if actionDunk:
                        print ("Action frame detected ", startFrame, actionDunk)
                    else:
                        print ("No action frame ", startFrame)
                        
        # Display the resulting frame
        cv2.imshow('Video out', frame)
        outVideo.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    outVideo.release()
    

if __name__== "__main__":
    exec(sys.argv[1])
    