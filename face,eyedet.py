


# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:50:04 2020

@author: Mitran
"""


import numpy as np
import cv2

facedetect=cv2.CascadeClassifier("./Haarcascades/face.xml")
eyedetect=cv2.CascadeClassifier("./Haarcascades/eye.xml")

inputimg=cv2.imread('./images/trump.jpg',0)
colorimg=cv2.imread('./images/trump.jpg')
face=facedetect.detectMultiScale(inputimg,1.3,5)

if(face is ()):
     cv2.putText(inputimg,'no face',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2),
   
    
    
    
else:
    for (fx,fy,fh,fw) in face:
        roi=inputimg[fx:fx+fh,fy:fw]
        roi_col=colorimg[fx:fx+fh,fy:fw]
        eye=eyedetect.detectMultiScale(roi)
        for (ex,ey,eh,ew) in eye:
           print('eye')
           cv2.rectangle(roi_col,(ex,ey),(ex+eh,ey+ew), (0,0,255), 3)
   
    cv2.rectangle(colorimg,(fx,fy),(fx+fh,fy+fw), (0,0,255), 3)
  # cv2.rectangle(inputimg,(face[0][0],face[0][1]),(face[0][0]+face[0][2],face[0][1]+face[0][3]), (0,0,255), 3)
    

cv2.imshow('sketch', colorimg)
    
cv2.waitKey()
       
cv2.destroyAllWindows()



