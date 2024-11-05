import streamlit as st
from PIL import Image
import joblib
import numpy as np
import cv2

# Load the model
model = joblib.load('C:/Users/HP/Desktop/Boadi/breast_cancer_model.pkl')

# Page 1 - Project Overview
def page_project():
    st.title("Breast Cancer Detection Project")
    image1 = Image.open("C:/Users/HP/Desktop/th.jpg")
    image2 = Image.open("C:/Users/HP/Desktop/th (1).jpg")
    st.image(image1, use_column_width=True)
    st.image(image2, use_column_width=True)
    st.write("""
    This project aims to develop a predictive model for breast cancer classification 
    using machine learning techniques. The model is designed to assist in the early 
    detection of breast cancer by analyzing various diagnostic features.This project aims to develop a machine learning model for detecting breast cancer using MRI scans. 
    It utilizes various algorithms, including Support Vector Machine and k-Nearest Neighbors, to accurately predict the presence of cancer.
    """)
    st.write("""
        The workflow involves:
        - Problem definition
        - Loading and analyzing the dataset
        - Evaluating machine learning algorithms
        - Standardizing the data
        - Tuning hyperparameters
        - Finalizing the model
    """)

# Page 2 - Breast Cancer Information
def page_breast_cancer():
    st.title("Understanding Breast Cancer")
    image = Image.open("C:/Users/HP/Desktop/th (2).jpg")
    st.image(image, use_column_width=True)
    st.write("""
    Breast cancer occurs when cells in the breast grow uncontrollably. 
    It is essential to understand the causes, effects, and prevention methods to combat this disease effectively. Breast cancer is a malignant tumor that develops from breast cells. It's one of the 
    most common cancers affecting women worldwide, but men can also develop breast cancer.
    
    **Causes**: 
    - Genetic factors
    - Hormonal factors
    - Lifestyle choices
    
    **Effects**: 
    - Physical and emotional impact
    - Changes in body image
    - Risk of metastasis

    **Prevention**: 
    - Regular screening
    - Healthy lifestyle choices
    - Genetic testing if at risk
    """)

# Page 3 - Upload MRI Scan for Prediction
def page_upload():
    st.title("Upload MRI Scan for Prediction")
    image = Image.open("C:/Users/HP/Desktop/th (3).jpg")
    st.image(image, use_column_width=True)

    uploaded_file = st.file_uploader("Choose an MRI scan...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Load the image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Preprocess the image for the model (add your preprocessing steps here)
        img = cv2.resize(img, (224, 224))  # Example resizing, change as needed
        img_array = np.array(img).reshape(1, -1)  # Reshape for model input

        # Make a prediction
        prediction = model.predict(img_array)
        if prediction[0] == 1:
            st.write("Prediction: **Cancer Detected**")
        else:
            st.write("Prediction: **No Cancer Detected**")

# Page 4 - Biography
def page_biography():
    st.title("Biography")
    image = Image.open("C:/Users/HP/Desktop/download.jpg")
    st.image(image, use_column_width=True)
    st.write("""
    I am an industrial engineering graduate with a keen interest in machine learning 
    and its applications in healthcare. My final year project focused on developing 
    a poultry defeathering machine, and I have participated in various projects related 
    to technology and engineering.
    """)
    st.write("Contact: 0593140049")
    st.write("Email: nanakofi8430@gmail.com")
    st.write("GitHub: [1nkz-coder](https://github.com/1nkz-coder)")

# Create the sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Project Overview", "Breast Cancer Information", "Upload MRI Scan", "Biography"))

# Display the selected page
if page == "Project Overview":
    page_project()
elif page == "Breast Cancer Information":
    page_breast_cancer()
elif page == "Upload MRI Scan":
    page_upload()
elif page == "Biography":
    page_biography()
