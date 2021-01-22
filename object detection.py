# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:32:16 2020

@author: Mitran
"""
def orbdetector(cropimg,image):
    #first convert htm to gray for easy comparison
    
    img1=cv2.cvtColor(cropimg,cv2.COLOR_RGB2GRAY)
    img2=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB(1000, 1.2)
   
        # Detect keypoints of original image,
    kp = orb.detect(img1,None)

    (kp1, des1) = orb.compute(img1, kp)
  
        # Detect keypoints of rotated image,
    kp = orb.detect(img2,None)
    (kp2, des2) = orb.compute(img2, kp)
   
    print(kp1,kp2,des1,des2)
        # Create matcher ,
      
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
        # Do matching,
    matches = bf.match(des1,des2)
    
        # Sort the matches based on distance.  Least distance,
        # is better,
    matches = sorted(matches, key=lambda val: val.distance)
   
    
    return len(matches)
import cv2
cap=cv2.VideoCapture(0)
inputimg=cv2.imread('./images/tobago.jpg')
sample = open('nsample.txt', 'w') 

  
while True:
    ret,frame=cap.read()
    """ pos=frame.shape
    
    posx=pos[0]
    posy=pos[1]
    frame=cv2.flip(frame, 1)
    cv2.rectangle(frame, (int(posx/2+100),int(posy/2-100)), (int(posx/2+300),int(posy/2+100)), (0,0,255),3)
   
    croppedportion=frame[int(posx/2+100):int(posy/2-100), int(posx/2+300):int(posy/2+100)]
   #flip vertically
   
   
    #matches=siftdetector(croppedportion,inputimg)
    cv2.putText(frame, str(2), (int(posx/2+200),int(posy/2-130)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0))"""
    print(frame)
    height= frame.shape[0]
    width=frame.shape[1]
    
    
        # Define ROI Box Dimensions (Note some of these things should be outside the loop),
    top_left_x = int(width / 3)
    top_left_y = int((height / 2) + (height / 4))
    bottom_right_x = int((width / 3) * 2)
    bottom_right_y = int((height / 2) - (height / 4))
      
        # Draw rectangular window for our region of interest,
    cv2.rectangle(frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y), (0,0,255), 3)
     
        # Crop window of observation we defined above,
    cropped = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]
   
        # Flip frame orientation horizontally,
    frame = cv2.flip(frame,1)
    matches = orbdetector(cropped, inputimg)
    # For new images or lightening conditions you may need to experiment a bit ,
        # Note: The ORB detector to get the top 1000 matches, 350 is essentially a min 35% match,
    threshold = 30
        
        # If matches exceed our threshold then object has been detected,
    if (matches > 217) and (matches<224):            
            cv2.putText(frame,'Object Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2),
            print(matches)
            
    #cv2.putText(frame, str(2), (150,150), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0))
    cv2.imshow('sketch', frame)
    
    if cv2.waitKey(1)==13:
        break
    
cap.release()

cv2.destroyAllWindows()
  
    