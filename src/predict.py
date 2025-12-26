# src/predict.py

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load trained model
model = load_model("models/digit_cnn.h5")

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Invert colors (MNIST is white digit on black background)
    image = ImageOps.invert(image)

    # Convert to numpy array
    image = np.array(image)

    # Normalize
    image = image / 255.0

    # Reshape for CNN
    image = image.reshape(1, 28, 28, 1)

    return image

def predict_digit(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    digit = np.argmax(prediction)
    return digit
