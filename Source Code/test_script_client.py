import matplotlib.pyplot as plt
import requests
import cv2
import os

#Emotion_reply={}
def getEmotionWithImage():
    img=cv2.imread('./image_and_results/Navya/joy.jpeg')

    cv2.imshow("frame",img)
    cv2.waitKey(0)
    
    url = 'http://127.0.0.1:5000/Emotion'
    my_img = {'image': open('./image_and_results/Navya/joy.jpeg', 'rb')}
    r = requests.post(url, files=my_img)

    # convert server response into JSON format.
    Emotion_reply=r.json()
    print(Emotion_reply)
    plotEmotionWithPercent(Emotion_reply)
    #print(r)
    
    cv2.destroyAllWindows() 


#-------------------------Ploting the Emotion-----------------------

def plotEmotionWithPercent(Emotion_reply):
    # labels for bars
    tick_label = list(Emotion_reply['emotion'].keys())

    # x-coordinates of left sides of bars 
    #left = list(range(1,len(tick_label)+1))
    
    # heights of bars
    height=[]
    for emval in tick_label:
        height.append(Emotion_reply['emotion'][emval])

    #import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    #total = 90000
    langs = tick_label
    langs_users_num = height

    #percent = langs_users_num/total*100

    new_labels = [i+'  {:.2f}%'.format(j) for i, j in zip(langs, langs_users_num)]

    plt.barh(langs, langs_users_num, color='lightskyblue', edgecolor='blue')
    plt.yticks(range(len(langs)), new_labels)
    plt.tight_layout()

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('Dominant Emotion - '+Emotion_reply['dominant_emotion'])
    ax.axes.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", left=False)
    plt.show()

def getEmotionWithVideo():

    file_name=""
    video_capture = cv2.VideoCapture(0) #replace with 0 for first camera
    while True:
        ret, frame = video_capture.read()
##        frame = cv2.resize(frame,(800,500))
        frame = cv2.putText(frame, 'Press \"Q\" to detect Emotion', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1,(0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            file_name=os.getcwd()+"/detect_im.jpg"
            cv2.imwrite(file_name,frame)
            break
#--------------------Send image to API------------------------
    url = 'http://127.0.0.1:5000/Emotion'
    my_img = {'image': open(file_name, 'rb')}
    r = requests.post(url, files=my_img)

    # convert server response into JSON format.
    Emotion_reply=r.json()
    print(Emotion_reply)
    plotEmotionWithPercent(Emotion_reply)

#-----------------Wait for user to press any key--------------------
    cv2.waitKey(0)
    video_capture.release()
    cv2.destroyAllWindows() 
    

#---------------append # in front-----------------------

if __name__=="__main__":
    getEmotionWithImage()
    #getEmotionWithVideo()
