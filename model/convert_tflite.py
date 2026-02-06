import tensorflow as tf

model = tf.keras.models.load_model('food_classifier.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('food_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Converted and saved as food_classifier.tflite")
