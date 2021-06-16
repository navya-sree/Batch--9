
from flask import Flask,render_template,request,flash,url_for,redirect,jsonify
from werkzeug.utils import secure_filename
import E_model_loader
import os

###############################################
#l=learner.load_learner("./models/level1.pth") #
###############################################

app=Flask(__name__)
app.secret_key = 'h432hi5ohi3h5i5hi3o2hi'

#create a route
'''@app.route('/')
def home():
    return render_template('index.html')'''
Emotion={}
@app.route('/Emotion',methods=['GET','POST'])
def result():
    if request.method == 'POST':
        img = request.files['image']
        path = os.getcwd()+"/detect_im.jpg"
        img.save(path)

        Emotion=E_model_loader.detect_emotion(path)
        print(Emotion)
        #return jsonify({'msg': 'success', 'size': [img.width, img.height]})
        return jsonify(Emotion)
    else:
        return jsonify({"err": "Only for API purporse"})
        #return jsonify(Emotion)

'''@app.route('/model')
def model():
    return render_template('model.html')'''

####################################

