import os
import cv2
import json
import thread
import numpy as np 
import pandas as pd
from fastai import *
import seaborn as sns
from fastai.vision import *
import matplotlib.pyplot as plt
from PIL import Image as PImage 
from sklearn.metrics import auc,roc_curve


class DeepFace:
    def __init__(self):
        cls.path= Path('AzGVBSGasjs123345-privateapi/GlobalDirectory/Fer2013_dataset/')
        cls.csv_file='AzGVBSGasjs123345-privateapi/GlobalDirectory/icml_face_data.csv')
        cls.class_names = {1: "sad", 2: "angry", 3: "surprise", 4: "fear",
                5: "happy", 6: "disgust", 7: "neutral"}
        cls.class_numbers = {"sad": 1, "angry": 2, "surprise": 3, "fear": 4,
                "happy": 5, "disgust": 6, "neutral": 7}
        cls.label_percentage = df.label.value_counts() / df.shape[0]
        cls.class_index = [class_names[idx] for idx in label_percentage.index.values]

    def train(self,Explicit_param,learner):
        tfms=get_transforms(flip_vert=True, max_warp=0.)
        data = (ImageList.from_folder(path)
        .split_by_rand_pct()
        .label_from_folder()
        .transform(tfms, size=150)
        .databunch(num_workers=2, bs=32))
        
        # Creating your Learner Model
        learner= cnn_learner(data, models.resnet34, metrics=[accuracy], model_dir='AzGVBSGasjs123345-privateapi/GlobalDirectory/API/flask-catcher/')
        learner1= cnn_learner(data, models.resnet34, metrics=[accuracy], model_dir='AzGVBSGasjs123345-privateapi/GlobalDirectory/API/flask-catcher/')
        
        # Training the Data
        learner.lr_find()

        #plot test
        #learner.recorder.plot()

        # Train the model on 4 epochs of data at the default learning rate
        #learner.fit_one_cycle(4)

        ## Fit the model over 8 epochs
        lr=5e-3  ## uncomment this
        learner.fit_one_cycle(8, lr)   ## uncomment this

        learner.save('AzGVBSGasjs123345-privateapi/GlobalDirectory/API/flask-catcher/facial_expression_model_weights.h5')
        

        learner1.load_learner('AzGVBSGasjs123345-privateapi/GlobalDirectory/API/flask-catcher/facial_expression_model_weights.h5')
        
        #load the model per build test
        #learner.load('AzGVBSGasjs123345-privateapi/GlobalDirectory/API/flask-catcher/facial_expression_model_weights.h5')

        #unfreze all layer of cnn
        learner.unfreeze() ## uncomment this

        # Find the optimal learning rate
        learner.lr_find()
        #learner.recorder.plot()

        ## (trying second attempt)Scrutenzing the data to make the training more efficient
        # learner.fit_one_cycle(16, slice(5e-5,5e-4))  # un comment this

        #save the model
        learner.save('weights_best_'+str(iter)) ## uncomment this
        learner.load_learner('weights_best_'+str(iter))

        #validate the model 
        interp = ClassificationInterpretation.from_learner(learner)
        interp1 = ClassificationInterpretation.from_learner(learner1)

        # intrepting most confused
        self.confused = interp.most_confused()
        self.confused1 = interp1.most_confused()

        for c,c1 in self.confused, self.confused1:
            conf+=c[3];conf1+=c1[3]
        
        # Call back loop for RNN
        conf > conf1 ?
            learner.save('AzGVBSGasjs123345-privateapi/GlobalDirectory/API/flask-catcher/facial_expression_model_weights.h5')
            :learner1.save('AzGVBSGasjs123345-privateapi/GlobalDirectory/API/flask-catcher/facial_expression_model_weights.h5')

        preds, lb=learner.get_preds()

        #  ROC curve
        fpr, tpr, thresholds = roc_curve(lb.numpy(), preds.numpy()[:,1], pos_label=1)

        #  ROC area
        pred_score = auc(fpr, tpr)

        return ("Done with training and laved in AD")
    
    def analyze(self, image, action):
        if 'train' in self.action:
            return "Not for training purporse"
        if 'emotion' in self.action:
            learner= cnn_learner(data, models.resnet34, metrics=[accuracy])
            learner.load('AzGVBSGasjs123345-privateapi/GlobalDirectory/API/flask-catcher/facial_expression_model_weights.h5')
            
            # Convert the input image to jpeg
            p="AzGVBSGasjs123345-privateapi/GlobalDirectory/API/flask-catcher/input_Image.jpeg"
            read = cv2.imread(t)
            cv2.imwrite(p,read)#,[int(cv2.IMWRITE_JPEG_QUALITY), 200])

            # Load the image as Pillow pimage
            test_image=plt.imread('AzGVBSGasjs123345-privateapi/GlobalDirectory/API/flask-catcher/input_Image.jpeg')
            #convert the image to np array and pridict
            frame = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
            pil_im = PImage.fromarray(frame) 
            x = pil2tensor(pil_im ,np.float32)
            preds_num = learner.predict(Image(x))[2].numpy()

            #Explicit calling train object through thread for LSTM
            thread.start_new(self.train(preds_num,learner))

            # create a json with corresponding prediction
            Emotion={}
            for index in range(1,len(preds_num)+1):
                Emotion[cls.class_names[index]]=preds_num[index]
            return json.dumps(Emotion,indent=4)
        return json.dumps({'err':'Improper parameter or token passed'},indent=4)
            


    


