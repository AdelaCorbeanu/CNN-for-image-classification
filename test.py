import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

# Read the metadata file
test_data = pd.read_csv('test.csv')

# Extract the image file names from the metadata
test_image_names = test_data['Image']

# Convert labels to categorical values
label_encoder = LabelEncoder()

# Prepare the image data
def preprocess_image(image_path):
    # Load and preprocess the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (64, 64))
    image = tf.keras.applications.resnet.preprocess_input(image)
    return image

# Load the saved model
saved_model = tf.keras.models.load_model('model_checkpoint.h5')

# Predict the test dataset
test_dataset = tf.data.Dataset.from_tensor_slices(test_image_names)
test_dataset = test_dataset.map(lambda x: tf.strings.join(['test_images/', x]))
test_dataset = test_dataset.map(preprocess_image)
test_dataset = test_dataset.batch(64)

predictions = saved_model.predict(test_dataset)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Decode class labels using the label encoder
predicted_labels = label_encoder.inverse_transform(predicted_labels)

# Create a DataFrame with the image names and predicted labels
results = pd.DataFrame({'Image': test_image_names, 'Class': predicted_labels})

# Save the results to a CSV file
results.to_csv('predictions.csv', index=False)
