#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Mar  4 23:18:37 2019

@author: fubao
"""
import socket
import numpy as np
import cv2


def recvData(sc, count):
    buf = b''
    while True:
        newbuf = sc.recv(count)
        print ("new buf received:", type(newbuf))
        if not newbuf: 
            print ("new buf none:")
            break
        buf += newbuf
        #count -= len(newbuf)
    return buf


def socketServerSetup():
    '''
    start a server here
    '''
    
    clientIp = "localhost"            # 192.168.0.52  192.168.0.53
    port = 1234
    sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         # TCP
    sc.bind((clientIp, port))
    sc.listen(True)
    conn, addr = sc.accept()
    
    #length = recvData(conn, 16)
    
    byteData = recvData(conn, 1024)  # int(length))
    if byteData is not None:
        data = np.fromstring(byteData, dtype='uint8')
    
        print ("image: ", type(data), len(data))
    
        decimg=cv2.imdecode(data,1)
        cv2.imwrite('Server_Rev.jpg', decimg)
        
    sc.close()
 
    
socketServerSetup()