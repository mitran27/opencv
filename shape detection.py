import cv2 
import numpy as np
"""
Image thresholding is a simple, yet effective, way of partitioning an image into a foreground and background. This image analysis technique is a type of image segmentation that isolates objects by converting grayscale images into binary images."""

inputimg = cv2.imread('./images/someshapes.jpg')
gray=cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)



ret,threshold=cv2.threshold(gray, 127, 255, 1)
""" 255 is the maximum value if any of the point grey scale value is greater than threshold value it converted to its maximum value"""
contours,hierachy=cv2.findContours(threshold.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(len(contours))
for cnt in contours:
    #initially approximate the contours polygon becz some noises can change the shape
    
    
    """approxPolyDP() allows the approximation of polygons, so if your image contains polygons, they will be quite accurately detected, combining the usage of cv2. findContours and cv2. approxPolyDP."""
    apx=cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True), True)
    #known as Ramer–Douglas–Peucker algorithm
    
    """ first one is image second one ellipson value it takes the point hisghest range which is has highest perpendicualr distance the plane if that distance is greater the ellipson it splits into two parts and call the recursive function to its two parts (one on leftt an one on right)  if its less than ellipson it returns as a single value

          Arc length is the distance between two points along a section of a curve
"""  
    if(len(apx)==3):
        shapename='Triangle'
        cv2.drawContours(inputimg, [cnt], 0, (0,255,0),-1)
        #shape name has to be placed in the center 
        point=cv2.moments(cnt)#take all the point
        """In image processing, computer vision and related fields, an image moment is a certain particular weighted average of the image pixels' intensities, or a function of such moments, usually chosen to have some attractive property or interpretation. Image moments are useful to describe objects after segmentation"""
        
        
       
        posx=int(point['m10']/point['m00'])
        posy=int(point['m01']/point['m00'])
        
        cv2.putText(inputimg, shapename, (posx,posy), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,0))
    elif(len(apx)==4):
        #to determine rectangle or square find its height and width
        x,y,h,w=cv2.boundingRect(cnt)
        if(abs(h-w)<=int(h*0.15)):
             shapename='square'
             cv2.drawContours(inputimg, [cnt], 0, (0,255,0),-1)       
             point=cv2.moments(cnt)       
             posx=int(point['m10']/point['m00'])
             posy=int(point['m01']/point['m00'])
        
             cv2.putText(inputimg, shapename, (posx,posy), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,0))
        else:
            shapename='rect'
            cv2.drawContours(inputimg, [cnt], 0, (0,255,0),-1)
            point=cv2.moments(cnt)
            posx=int(point['m10']/point['m00'])
            posy=int(point['m01']/point['m00'])
        
            cv2.putText(inputimg, shapename, (posx,posy), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,0))
    elif(len(apx)==10):
            shapename='star'
            cv2.drawContours(inputimg, [cnt], 0, (0,255,0),-1)
            point=cv2.moments(cnt)
            posx=int(point['m10']/point['m00'])
            posy=int(point['m01']/point['m00'])
        
            cv2.putText(inputimg, shapename, (posx,posy), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,0))
    else:
            shapename='circle'
            cv2.drawContours(inputimg, [cnt], 0, (0,255,0),-1)
            point=cv2.moments(cnt)
            posx=int(point['m10']/point['m00'])
            posy=int(point['m01']/point['m00'])
        
            cv2.putText(inputimg, shapename, (posx,posy), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,0))
    print((len(cnt)),len(apx))
        
    
cv2.imshow('Hello World', inputimg)
cv2.waitKey(0)



cv2.destroyAllWindows()