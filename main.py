import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model("modul1.keras")

# Convert it to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("modul1.tflite", "wb") as f:
    f.write(tflite_model)

