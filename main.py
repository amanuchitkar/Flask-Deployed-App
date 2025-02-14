import tensorflow as tf

# Load the original Keras model
model = tf.keras.models.load_model("modul1.keras")

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable optimization
tflite_model = converter.convert()

# Save the optimized TFLite model
with open("modul1.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully.")
