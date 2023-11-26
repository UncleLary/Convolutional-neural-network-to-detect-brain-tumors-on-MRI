from dotenv import load_dotenv
from keras.models import load_model
import os
from flask import Flask, request, render_template
from PIL import Image
import numpy as np


# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from flask import Flask, request, render_template
# from werkzeug.utils import secure_filename

load_dotenv()
# pathToNetwork = os.getenv(r"pathToPresenceDetectingNetwork")
pathToNetwork =  r"C:\Users\lakie\Desktop\Praca inżynierska\CNN_BrainMRI\Networks\tumorPresenceDetector10Epochs.keras"
model = load_model(pathToNetwork)
print("Model loaded")

app = Flask(__name__)

def displayPrediction(number):
    if number == 0:
        return "No neoplastic lesions were found"
    elif number == 1:
        return "Neoplastic lesions were found"
    else:
        return "Sorry, something went wrong"

def getDetectionResult(inputImage):
    inputImage = Image.open(inputImage)
    inputImage = inputImage.convert('RGB')
    inputImage = inputImage.resize((64, 64))  # 64 == current input image size
    inputImage = np.array(inputImage)
    inputImage = np.expand_dims(inputImage, axis=0)
    result=model.predict(inputImage)
    result = int(result[0][1])
    return result

@app.route('/', methods=['GET'])
def index():
    # render_template('index.html')
    print("Get_method")


@app.route('/predict', methods=['GET','POST'])
def predictInputImage():
    if request.method == 'POST':
        inputImage = request.files['file']

        detectionResult = getDetectionResult(inputImage)
        predictionToDisplay = displayPrediction(detectionResult)
        return predictionToDisplay
    return None

if __name__ == '__main__':
    app.run(debug=True)



# model =load_model('BrainTumor10Epochs.h5')
# print('Model loaded. Check http://127.0.0.1:5000/')
#
#
#
# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')
#
#
# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         f = request.files['file']
#
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)
#         value=getResult(file_path)
#         result=get_className(value)
#         return result
#     return None
