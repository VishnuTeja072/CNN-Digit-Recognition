import tensorflow as tf
from src.model import build_cnn

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize & reshape
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build & train
model = build_cnn()
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Save model
model.save("models/digit_cnn.h5")
print("✅ Model saved at models/digit_cnn.h5")

