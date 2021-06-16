 
from deepface import DeepFace
import cv2
#import matplotlib.pyplot as plt
#import requests

#img1 = cv2.imread("./Navya/Angry.jpeg")
#plt.imshow(img1[:,:,::-1])
#plt.show()
def detect_emotion(img1):
    img1=cv2.imread(img1)
    result = DeepFace.analyze(img1,actions = ['emotion'])
    print(result)
    return result

#img1 = cv2.imread("./image_and_results/Navya/Angry.jpeg")
#a=detect_emotion(img1)


