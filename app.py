import os
import logging
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf

# Set up logging to debug issues
logging.basicConfig(level=logging.DEBUG)

# Load disease and supplement information CSV files
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the Keras model
model = tf.keras.models.load_model('modul1.keras')

# Prediction function
def model_prediction(test_image):
    try:
        # Load the image and preprocess it
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        predictions = model.predict(input_arr)
        
        # Log predictions to debug
        logging.debug(f"Predictions: {predictions}")
        
        # Return the index of the predicted class (highest probability)
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
        # Retrieve image from the request
        image = request.files['image']
        
        if image:
            try:
                # Create a filename and save the image to the static/uploads directory
                filename = image.filename
                file_path = os.path.join('static/uploads', filename)
                image.save(file_path)

                # Perform prediction
                pred = model_prediction(file_path)

                if pred is not None:
                    # Log the prediction index for debugging
                    logging.debug(f"Prediction Index: {pred}")

                    # Fetch disease info from CSV based on the prediction index
                    title = disease_info['disease_name'][pred]
                    desc = disease_info['description'][pred]
                    prevent = disease_info['Possible Steps'][pred]
                    image_url = disease_info['image_url'][pred]

                    # Fetch supplement info from CSV based on the prediction index
                    sname = supplement_info['supplement name'][pred]
                    simage = supplement_info['supplement image'][pred]
                    buy_link = supplement_info['buy link'][pred]

                    # Return the result to the submit.html page, including the filename
                    return render_template('submit.html', 
                       pred=pred, 
                       title=title, 
                       desc=desc, 
                       prevent=prevent, 
                       image_url=image_url, 
                       sname=sname, 
                       simage=simage, 
                       buy_link=buy_link,
                       filename=filename)

                else:
                    # Handle case where prediction fails
                    logging.error("Prediction failed")
                    return render_template('error.html', message="Prediction failed.")
            
            except Exception as e:
                logging.error(f"Error during image processing or prediction: {e}")
                return render_template('error.html', message="Error processing the image.")
        else:
            # Handle case where no image is uploaded
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
