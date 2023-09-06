#!/usr/bin/env python
# coding: utf-8
import  pandas as pd
import streamlit as st
import keras
from PIL import Image
import numpy as np
import  matplotlib as plt
import  requests

model_release= "https://github.com/iamanimesh11/mediPredict_-Disease-prediction-all-in-one/releases/download/brainTumorModel/brain_tumor_detection_model.h5"
def download_asset(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

download_asset(model_release, 'brain_tumor_detection_model.h5')


def show_project_overview_page():
    st.header("Introduction to the Brain tumor Disease Prediction Project")

    st.markdown("""
    Understand your brain tumor disease risk using advanced deep learning. Input your health data to receive personalized predictions and insights. With 98% accurate results, take charge of your health.
    """)

    st.header("Project Overview")

    st.markdown("""
    The project is structured into several key steps:
    """)

    st.subheader("1. Data Cleaning and Exploratory Data Analysis (EDA):")
    st.markdown("""
    -  collection of a dataset comprising MRI brain scan images. .
    - To prepare the data for analysis, extensive image preprocessing techniques were applied.
    -  This included resizing images to a consistent format, normalizing pixel values, and addressing any artifacts or noise present in the scans.
    """)

    st.subheader("2. EDA")
    st.markdown("""
    - An initial exploratory data analysis was performed to gain insights into the distribution of images, identifying any patterns or anomalies. Visualizations and statistical analysis may have been used to understand the composition of the dataset and to check for class imbalances.
    """)

    st.subheader("3. Model Selection and Building:")
    st.markdown("""
    - The core of the project involved selecting and developing a deep learning model to detect brain tumors from the MRI images. Convolutional Neural Networks (CNNs) were the primary choice due to their effectiveness in image classification tasks. Different CNN architectures and configurations were explored, and hyperparameter tuning was conducted to optimize model performance.
    """)

    st.subheader("4. Model Ensembling with Stacking:")
    st.markdown("""
    - To enhance transparency and trust in the model's predictions, techniques for model interpretability were employed. This included visualization of feature maps, heatmaps, and saliency maps to understand which regions of the images were most influential in making predictions.
    """)

    st.subheader("5. Model Deployment:")
    st.markdown("""
    - The final trained model was deployed as a user-friendly web application using Streamlit. This allowed users to upload their MRI brain scan images, and the model would provide real-time predictions regarding the presence of brain tumors. The deployment phase ensured accessibility to medical professionals and patients for practical use.
    """)

    # Data for model evaluation
    model_data = {
        'Model': [' DecisionTree', 'Logistic Regression ', 'K-Nearest Neighbour', 'Random Forest'],
        'Accuracy': [70.213, 70.4891, 64.844, 83.647554]
    }

def display():
    st.sidebar.title('Welcome to the Brain Tumour Predictior App')
    st.sidebar.write('This app uses a deep learning model to predict the likelihood of Brain Tumour disease.')
    with st.sidebar.expander("How it Works❓❓"):
        st.write(
            "1. **Input Health Data**: Upload MRI scan of brain and wait for model to work upon it.")
        st.write(
            "2. **Instant Prediction**:  state-of-the-art deep learning model analyzes your input and generates an instant prediction about your risk of brain tumor disease.")

    # Load the pre-trained model
    model = keras.models.load_model('brain_tumor_detection_model.h5')
    imcol, textcol,x = st.columns(3)

    textcol.header("Brain Tumor Prediction")

    image = Image.open("image/2f78bf0d8bb4800763136be686630334.png")

    new_size = (150, 150)  # Define the new size (width, height)
    resized_image = image.resize(new_size)

    # Display the resized image
    imcol.image(resized_image)

    # Function to make predictions
    def predict(image):
        img = Image.open(image)
        img = img.resize((128, 128))
        img = np.array(img)
        img = img.reshape(1, 128, 128, 3)
        res = model.predict(img)
        classification = np.where(res == np.amax(res))[1][0]
        return classification

    # Streamlit app title and description
    st.write("Upload an MRI brain scan image to detect brain tumors.")

    # File uploader widget for image input
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg"])

    # Function to display result text
    def names(number):
        if number == 0:
            return 'It\'s a Tumor'
        else:
            return 'No, It\'s not a tumor'

    # Display results when an image is uploaded
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        result = predict(uploaded_image)
        st.success(f"Prediction: {names(result)}")
