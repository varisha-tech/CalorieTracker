import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Load dataset (Food-101)
(ds_train, ds_val), ds_info = tfds.load(
    'food101',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True
)

IMG_SIZE = 224
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE


def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, tf.one_hot(label, ds_info.features['label'].num_classes)


train = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE).batch(
    BATCH).prefetch(AUTOTUNE)
val = ds_val.map(preprocess, num_parallel_calls=AUTOTUNE).batch(
    BATCH).prefetch(AUTOTUNE)

# Model
base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(
    ds_info.features['label'].num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=5)

model.save('food_classifier.h5')
print(" Model trained and saved as food_classifier.h5")
