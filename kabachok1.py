import cv2
import numpy as np  
img =cv2.imread('images/layout.jpg')
img=cv2.resize(img,(img.shape[1]*8//12, img.shape[0]*8//12))
wbimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
wbimg=cv2.Canny(wbimg,50,50)
cv2.imshow('result', wbimg)
cv2.waitKey(0)  
#cv2.imshow('Result', img)
#print(img.shape)
#cv2.waitKey(0)  
#cap =cv2.VideoCapture(0)
#cap.set(3,500)
#cap.set(4,500)

#while True:
 #   success, img= cap.read()
  #  cv2.imshow('Result',img)

   # if cv2.waitKey(0) & 0xFF == ord('q'):
    #    break 
