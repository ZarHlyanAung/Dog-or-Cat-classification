# import Libraries
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
# import imutils
# from imutils.perspective import four_point_transform
# from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input


# <---- functions start ---->

# loaded the model from other path
@st.cache_data
def loadModel():
    model = tf.keras.models.load_model(
        "version-nine-from-five.h5", compile=False)
    return model


model = loadModel()


# <--- factions end --->


st.markdown("<h3 style='color: #1A5F7A;'>Dog or Cat classification</h3>",
            unsafe_allow_html=True)
st.write("* no two animals in one pic, only one cat or dog input image is for good prediction")


# load file
uploaded_file = st.file_uploader("Upload Your Image", type=["png", "jpg"])

if uploaded_file is None:
    st.success("Please upload your image")

elif uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image Input')

    # Resize the image to 100x100
    resized = cv2.resize(np.array(image), (100, 100))
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis, ...]

    genrate = st.button('Genrate Prediction')
    if genrate:
        prediction = model.predict(img_reshape)
        if prediction >= 0.5:
            st.header("Dog")
        else:
            st.header("Cat")
