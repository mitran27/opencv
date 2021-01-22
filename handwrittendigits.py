# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 09:39:09 2020

@author: Mitran
"""


import numpy as np
import cv2
def findcenter(contour):
    
        M = cv2.moments(contour)
        return (int(M['m10']/M['m00']))
def cropimg(roi):
     #converts region of interest to a square with paddding so it can be easily resized
    #print('convert the img to a padded square image')
    #print(roi.shape)
    sha=roi.shape
    if(sha[0]==sha[1]):
        return(roi)
    else:
        #integer argument expected, got float so doubling it
        height=sha[0]
        width=sha[1]
        roi = cv2.resize(roi.copy(),(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        if(height>width):
            pad=int((height-width)/2)
            newimg=cv2.copyMakeBorder(roi, 0, 0, pad, pad, cv2.BORDER_CONSTANT,value=[0,0,0])
            #print(newimg.shape)
        else:
            pad=int((width-height)/2)
            newimg=cv2.copyMakeBorder(roi, pad, pad, 0, 0, cv2.BORDER_CONSTANT,value=[0,0,0])
            #print(newimg.shape)
           # cv2.copyMakeBorder(src, top, bottom, left, right, borderType)
        return newimg
def resize_to_pixel(dimensions, image):
    # This function then re-sizes an image to the specificied dimenions
    """
     buffer_pix = 4
    dimensions  = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]"""
   
    x=20
    buffer=x/5
    dime=x-buffer   
    dim = (int(dime), int(dime))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0,0,0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    p = int(x/10)
    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg   
    
# Let's take a look at our digits dataset
image = cv2.imread('images/digits.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
small = cv2.pyrDown(gray)#large value image are unable to display
cv2.imshow('digits', small)
cv2.waitKey(0)

# the data set is 50 rows 100 colums

dataset=np.array([np.hsplit(row,100)for row in np.vsplit(gray,50)])
#print(dataset.shape)
train=dataset[:,:].reshape(-1,400).astype(np.float32)
#test=dataset[:,70:100].reshape(-1,400).astype(np.float32)# -1 means all not negative one
#print(dataset[0][0].shape,train[0].shape)
#print(train.shape,test.shape)

k = [0,1,2,3,4,5,6,7,8,9]
train_labels = np.repeat(k,500)[:,np.newaxis]
#test_labels = np.repeat(k,150)[:,np.newaxis]
#print(train_labels.shape)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10).fit(train,train_labels.ravel()) 

"""

result=knn.predict(test)
match=(result==test_labels.ravel())
correctness=np.count_nonzero(match)
accuracy=(correctness/result.size)*100
print(accuracy)

"""


image = cv2.imread('images/numbers.jpg')


bawimg=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blurimg=cv2.GaussianBlur(bawimg, (5,5), 1)
edged = cv2.Canny(blurimg, 30, 150)
contours, hierachy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('numbers', edged)
cv2.waitKey(0)

full_number = []
contours = sorted(contours, key = findcenter, reverse = False)
for cnt in contours:
     (x, y, w, h) = cv2.boundingRect(cnt)
     
     cv2.drawContours(image, cnt, -1, (0,255,0), 3)
    
    
     #small cracks should not be considered so filter it
     if w >= 15 and h >= 50:
          roi = blurimg[y:y + h, x:x + w]
          #roi=cv2.GaussianBlur(roi, (5,5), 1)
          ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
          #cropperfect image
          convertedimg=cropimg(roi)
          # resize img to 20* 20 so that input can be given
          resizedimg=cv2.resize(convertedimg, (20,20),interpolation=cv2.INTER_CUBIC)#less accuracy 
          resizedimg=resize_to_pixel(20, resizedimg)# more accuracy
          #print(resizedimg.shape)
          number=knn.predict(resizedimg.reshape(-1,400))[0]
          cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
          cv2.putText(image, str(number), (x , y + 155),cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)
          #cv2.imshow("Contours", resizedimg)
          #cv2.waitKey(0)
cv2.imshow("Contours", image)
cv2.waitKey(0)



cv2.destroyAllWindows()