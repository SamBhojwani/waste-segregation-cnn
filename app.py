import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('waste_classifier.h5')

# Define class names
class_names = ['Organic', 'Recyclable']

# Streamlit UI
st.title('♻️ Waste Classifier')
st.write('Upload an image of waste and let AI predict if it is **Organic** or **Recyclable**!')

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    # Preprocess the image
    img_array = np.array(image)
    img = tf.image.resize(img_array, (224, 224))  # Resize to 224x224
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Prediction
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]

    st.success(f'Prediction: **{predicted_class}**')

# To run, use : streamlit run app.py