#Importing Essential Flask Libaries

from flask import Flask,render_template,url_for,request,redirect
from werkzeug.utils import secure_filename
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from keras.preprocessing import image
#import sys
import os
import glob
import re
import numpy as np
#Define the flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = "resnet.h5"


# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()


# Funcion to predicit the imagenet_utils
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0) #This function expands the array by inserting a new axis at the specified position

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        f = request.files['file']

         # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        # Make prediction
        preds = model_predict(file_path, model)

        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode # Decode will help to match the class index to imagent class and returnt the value
        result = str(pred_class[0][0][1])               # Convert to string
        return result #render_template('predict.html',result=result)

    return none





if (__name__) == '__main__':
    app.run(debug=True)
