import os
import tensorflow as tf
import numpy as np

# Set the relative path for the model
model_path = 'digit_recognition_model.h5'  # Use .keras or .h5 extension
num_classes = 10
# Load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Manipulating the data for consistent format
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)  # Always one-hot encode y_test

if os.path.exists(model_path):
    print("Loading existing model...")
    model = tf.keras.models.load_model(model_path)
else:
    print("Training new model...")
    input_shape = (28, 28, 1)

    # Manipulate training data
    x_train = x_train.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    print("Shape of training labels:", y_train.shape)

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, validation_split=0.2)

    # Save the model
    model.save(model_path)
    print("Model trained and saved at", model_path)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")
