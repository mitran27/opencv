


# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:50:04 2020

@author: Mitran
"""


import numpy as np
import cv2
cap=cv2.VideoCapture('C:/Users/Mitran/Desktop/projects/opencv/images/cars.avi')


  
while True:
    ret,frame=cap.read()

    bodydetect=cv2.CascadeClassifier("./Haarcascades/full.xml")
    cardetect=cv2.CascadeClassifier("./Haarcascades/car.xml")
    
    inputimg=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    colorimg=frame
    body=bodydetect.detectMultiScale(inputimg,1.3,5)
    car=cardetect.detectMultiScale(inputimg,1.3,5)
    print(len(car),len(body))

    for (fx,fy,fh,fw) in body:
           cv2.rectangle(colorimg,(fx,fy),(fx+fh,fy+fw), (0,0,255), 3)
    for (fx,fy,fh,fw) in car:
           cv2.rectangle(colorimg,(fx,fy),(fx+fh,fy+fw), (0,255,255), 3)
      

    cv2.imshow('sketch', colorimg)
    if cv2.waitKey(1)==13:
        break
    
cv2.waitKey()
       
cv2.destroyAllWindows()



