# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 11:56:40 2020

@author: Mitran
"""


import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = './faces/user/'
face_classifier = cv2.CascadeClassifier('Haarcascades/face.xml')
def changetraindata():
    print('ready to read training data')
    cap = cv2.VideoCapture(0)
    imgcount=0
    total_Count=200
    while True:
         ret, frame = cap.read()
         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
         image,face = detectface(gray)
        
  
         if face.size >1:    
             print('its a face')
             imgcount += 1
             face = cv2.resize(face, (200, 200))
            

        # Save file in specified directory with unique name
             file_name_path = './faces/user/' + str(imgcount) + '.jpg'
             cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
             cv2.putText(image, str(imgcount), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
             cv2.imshow('Face Cropper', image)
         else:
             print('nooo')
         if cv2.waitKey(1) == 13 or imgcount>=total_Count: #13 is the Enter Key
               break
    
    
def detectface(image):
          face=face_classifier.detectMultiScale(image, 1.3, 5)
          if face is ():              
              return image,np.ones((1,1))
          else:
              x,y,w,h=face[0]
              roi=image[y:y+h, x:x+w]
              roi=cv2.resize(roi,(200,200))
              img=cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0))
              
              return img,roi
def traintheface():
    userimages=[]
    Training_Data=[]# if there is training data there must be labels
    labels=[]
    print('training model')
    for f in listdir(data_path):
        if (isfile(join(data_path,f))):
            userimages.append(f)
    for i,f in enumerate(userimages):
        imgpath=join(data_path,f)
        image=cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(image)
        labels.append(i)
   
    labels=np.asarray(labels,dtype=np.int32)
   
        
    model =cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(labels))
    print("Model trained sucessefully")     
    return model       

whichface=int(input('enter new face(1) or old face(0)'))
if(whichface==1):
    changetraindata()
    
model=traintheface()
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    image,face = detectface(gray)
    allow=False
      
    #print(face.size)
    if face.size >1:    
    
        results = model.predict(face)
        confidence=int( 100 * (1 - (results[1])/400) )
        print(results)
        if(confidence>75):
            cv2.putText(image, "unlocked", (20,20), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255))
            print('unlocked')
            allow=True
            break
           
        else:
            cv2.putText(image, "locked", (20,20), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255))
            print('locked')
    else:
         cv2.putText(image, "mooonja katra", (20,20), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255))
         print('mooonja katra')
    cv2.imshow('Face Cropper', image)
        
       
       
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
     
cap.release()
if(allow):
   input = cv2.imread('./images/input.jpg')
   cv2.imshow('your gift', input)
   cv2.waitKey()  
    

cv2.destroyAllWindows()     