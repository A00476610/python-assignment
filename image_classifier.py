import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained model
model_path = 'mnist_model.keras'
model = tf.keras.models.load_model(model_path)


def predict_digit(img):
    """Preprocesses the image and predicts the digit."""
    # Convert the image to grayscale and resize it to 28x28
    # img = img.convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert the image to a numpy array and normalize
    img_array = np.array(img)
    # img_array = img_array / 255.0
    # img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    predictions = model.predict(img_array)
    return np.argmax(predictions)



def predict_digit(img_array):
    # """Preprocesses the image and predicts the digit."""
    # # Convert the image to grayscale and resize it to 28x28
    # # img = img.convert('L')
    # img = img.resize((28, 28), Image.Resampling.LANCZOS)
    #
    # # Convert the image to a numpy array and normalize
    # img_array = np.array(img)
    # # img_array = img_array / 255.0
    # # img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    predictions = model.predict(img_array)
    return np.argmax(predictions)


def process_image(original_img):

    # Convert images with transparency to an opaque format
    if original_img.mode in ('RGBA', 'LA', 'P'):
        background_layer = Image.new("RGB", original_img.size, "WHITE")
        background_layer.paste(original_img,
                               mask=original_img.getchannel('A') if original_img.mode == 'RGBA' else original_img)
        processed_img = background_layer
    else:
        processed_img = original_img.convert('RGB')

    # Convert to grayscale and apply inversion for digit clarity
    processed_img = ImageOps.grayscale(processed_img)
    processed_img = ImageOps.invert(processed_img)

    # Calculate the new size keeping aspect ratio constant
    aspect = min(28 / processed_img.width, 28 / processed_img.height)
    new_dimensions = (int(processed_img.width * aspect), int(processed_img.height * aspect))
    processed_img = processed_img.resize(new_dimensions, Image.Resampling.LANCZOS)

    # Place the processed image onto a centered canvas
    canvas = Image.new('L', (28, 28), 'white')
    paste_coords = ((28 - new_dimensions[0]) // 2, (28 - new_dimensions[1]) // 2)
    canvas.paste(processed_img, paste_coords)

    # Prepare the image array for the model
    img_array = np.array(canvas).astype(np.float32) / 255.0
    img_array = img_array.reshape(-1, 28, 28, 1)  # Ensuring proper dimensions for the model

    return img_array



def main():
    st.title('Digit Recognition App')
    st.write("Upload an image of a single digit, and the model will predict the digit.")

    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict and display the results
        if st.button('Predict'):
            result = predict_digit(image)
            st.write(f'The model predicts: {result}')
        else:
            st.write('Click the predict button to classify the digit.')


if __name__ == '__main__':
    main()


