
# Authenticate to access your GCS bucket
from google.colab import auth
auth.authenticate_user()

import tensorflow as tf
import os
import pickle

# Define parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# Function to process each image file path
def process_path(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    label_str = parts[-2]
    label = tf.cond(tf.equal(label_str, "human_art"), lambda: 1, lambda: 0)

    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

# Create dataset from GCS
file_patterns = ["gs://polygence_image_bucket/train/*/*.jpg", "gs://polygence_image_bucket/train/*/*.png"]
list_ds = tf.data.Dataset.list_files(file_patterns, shuffle=True)
ds = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Define CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(ds, epochs=2)

# --- Save model as a pickle file instead of .h5 ---
# Convert model to JSON + weights
model_config = model.to_json()
model_weights = model.get_weights()

# Save both into a .pkl file
model_data = {
    'config': model_config,
    'weights': model_weights
}

with open("art_classifier.pkl", "wb") as f:
    pickle.dump(model_data, f)

# Upload .pkl to your GCS bucket
#!gsutil cp art_classifier.pkl gs://polygence_image_bucket/models/art_classifier.pkl
