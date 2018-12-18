#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:21:59 2018

@author: fubao
"""
# opencv test


# use  Haar cascade classifiers to do detect human

# https://github.com/opencv/opencv/tree/master/data

# model: haarcascade_upperbody.xml;  haarcascade_fullbody.xml

# https://medium.com/@madhawavidanapathirana/https-medium-com-madhawavidanapathirana-real-time-human-detection-in-computer-vision-part-1-2acb851f4e55


import cv2
import sys
import os
from dataComm import loggingSetting
from dataComm import readVideo


def testImage(modelPath, inputImage):
    
    bodyCascade = cv2.CascadeClassifier(modelPath)
    print (" model path: ", modelPath)
    

    frame = cv2.imread(inputImage)
    cv2.imshow('Original Image', frame)

    height, width, channels = frame.shape
    print (width, height, channels)
    
    #imScale = cv2.resize(frame, (640,360)) # Downscale to improve frame rate
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    bodies = bodyCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(5, 5),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in bodies:
        print (" human detected: ")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('detectd human', frame)
    cv2.waitKey(0)
    
    
def detectHuman(modelPath, videoPath, outputVideoName):
    '''
    xml model pretrained 
    videoPath: input a vdideo
    
    '''
    faceCascade = cv2.CascadeClassifier(modelPath)
    print ("videoPath, model path: ", videoPath, modelPath)
    
    cap = readVideo(videoPath)
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    NUMFRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    print ('cam stat: %s, %s, %s, %s ', fps, WIDTH, HEIGHT, NUMFRAMES)
    
    # outputVideoName =  "UCF101_v_longJump_g01_c01_out.avi"   # "UCF101_v_BasketballDunk_g01_c01_out.avi"  # "humanRunning_JHMDB_output_001.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')   # MJPG
    outputVideoDir = os.path.join( os.path.dirname(__file__), '../output/')
    outVideo = cv2.VideoWriter(outputVideoDir + outputVideoName, fourcc, 5,  (int(WIDTH), int(HEIGHT)))
    
    
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print ("no frame exit here 1, total frames ", ret)
            break
        
        imScale = cv2.resize(frame,(640,360)) # Downscale to improve frame rate
        gray = cv2.cvtColor(imScale, cv2.COLOR_BGR2GRAY)
        print ("model pathssss: ", ret)
        

                
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=0,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
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
    exec(sys.argv[1])