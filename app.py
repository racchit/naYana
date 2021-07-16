import flask
from flask import Flask,render_template,url_for,request
import base64
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model


#Initialize the useless part of the base64 encoded image.
init_Base64 = 21

#Our dictionary
label_dict = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9'}


#Initializing the Default Graph (prevent errors)

# Use pickle to load in the pre-trained model.
model = load_model('vowels_numbers.h5')

#Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')

#First route : Render the initial drawing template
@app.route('/')
def home():
	return render_template('draw.html')



#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
        if request.method == 'POST':
                final_pred = None
                #Preprocess the image : set the image to 28x28 shape
                #Access the image
                draw = request.form['url']
                #Removing the useless part of the url.
                encoded_data = request.form['url'].split(',')[1]
                #Decoding
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imwrite('iamge.png', img)
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                #Resizing and reshaping to keep the ratio.
                resized = cv2.resize(gray_image, (50,50), interpolation = cv2.INTER_AREA)
                vect = np.asarray(resized, dtype="uint8")
                vect = vect.reshape(1, 50, 50, 1).astype('float32')
                #Launch prediction
                my_prediction = model.predict(vect)
                #Getting the index of the maximum prediction
                index = np.argmax(my_prediction[0])
                #Associating the index and its value within the dictionnary
                final_pred = label_dict[index]
        return render_template('results.html', prediction =final_pred)


if __name__ == '__main__':
	app.run(debug=True)

