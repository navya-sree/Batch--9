 
import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2 as cv
import os
from PIL import Image

def face(img):
  
    # Read the input image 
    #img = cv2.imread(image) 
    
    # Convert into grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Load the cascade 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') 
    
    # Detect faces 
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
    
    # Draw rectangle around the faces and crop the faces 
    for (x, y, w, h) in faces: 
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) 
        faces = img[y:y + h, x:x + w] 
        cv2.imshow("face",faces) 
        cv2.imwrite('face.jpg', faces) 
    img=cv2.imread('face.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Load the cascade 
    face_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
    
    # Detect faces 
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
    
    # Draw rectangle around the faces and crop the faces 
    for (x, y, w, h) in faces: 
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) 
        faces = img[y:y + h, x:x + w] 
        #cv2.imshow("face",faces) 
        #cv2.imwrite('face.jpg', faces) 
        
    # Display the output 
    #cv2.imwrite('detcted.jpg', img) 
    #cv2.imshow('img', img) 
    #cv2.waitKey()
    return img

img1 = cv.imread('./Priya/angry.jpeg')
img2 = cv.imread('./Nisha/angry.jpg')
img3 = cv.imread('./Navya/Angry.jpeg')

dst1 = cv.fastNlMeansDenoisingColored(img1,None,10,10,7,21)
img1=face(dst1)
img1 = cv.imread('face.jpg')
dst2 = cv.fastNlMeansDenoisingColored(img2,None,10,10,7,21)
img2=face(dst2)
img2 = cv.imread('face.jpg')
dst3 = cv.fastNlMeansDenoisingColored(img3,None,10,10,7,21)
img3=face(dst3)
img3 = cv.imread('face.jpg')


plt.subplot(141),plt.imshow(dst1)
plt.subplot(142),plt.imshow(dst2)
plt.subplot(143),plt.imshow(dst3)

plt.show()
