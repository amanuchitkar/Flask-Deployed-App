import os
import logging
import numpy as np
import pandas as pd
import tensorflow.lite as tflite
from flask import Flask, render_template, request
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load CSV files
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="modul1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prediction function
def model_prediction(test_image):
    try:
        image = Image.open(test_image).resize((128, 128))
        input_arr = np.array(image, dtype=np.float32) / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
        
        interpreter.set_tensor(input_details[0]['index'], input_arr)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        logging.debug(f"Predictions: {predictions}")
        return np.argmax(predictions)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        
        if image:
            try:
                filename = image.filename
                file_path = os.path.join('static/uploads', filename)
                image.save(file_path)

                pred = model_prediction(file_path)
                
                if pred is not None:
                    logging.debug(f"Prediction Index: {pred}")
                    title = disease_info['disease_name'][pred]
                    desc = disease_info['description'][pred]
                    prevent = disease_info['Possible Steps'][pred]
                    image_url = disease_info['image_url'][pred]
                    sname = supplement_info['supplement name'][pred]
                    simage = supplement_info['supplement image'][pred]
                    buy_link = supplement_info['buy link'][pred]

                    return render_template('submit.html', 
                        pred=pred, title=title, desc=desc, 
                        prevent=prevent, image_url=image_url, 
                        sname=sname, simage=simage, 
                        buy_link=buy_link, filename=filename)
                else:
                    logging.error("Prediction failed")
                    return render_template('error.html', message="Prediction failed.")
            except Exception as e:
                logging.error(f"Error during image processing or prediction: {e}")
                return render_template('error.html', message="Error processing the image.")
        else:
            return render_template('error.html', message="No image uploaded.")
    
    return render_template('submit.html')

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), 
                           disease=list(disease_info['disease_name']), 
                           buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
