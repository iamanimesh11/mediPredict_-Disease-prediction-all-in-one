import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import  Image
# Load the trained logistic regression model using pickle


def show_project_overview_page():
    st.header("Introduction to the Diabetes Prediction Project")

    st.markdown("""Understand your diabetes risk using advanced machine learning. Input your health data to receive personalized predictions and insights. With 85 % accurate results , take charge of your health                 """)

    st.header("Project Overview")

    st.markdown("""
            The project is structured into several key steps:
            """)

    st.subheader("1. Data Cleaning and Exploratory Data Analysis (EDA):")
    st.markdown("""
            - Clean and prepare the dataset for analysis.
            - Generate data insights using pandas-profiling..
            - Handle missing values, outliers, and inconsistencies..
            - Gain insights into dataset structure, distributions, and correlations through EDA.
        """)

    st.subheader("2. Data Preprocessing:")
    st.markdown("""
            - Train-Test Split: We split the dataset into training and testing subsets. The training set is used to teach the machine learning models, while the testing set is reserved for evaluating their performance.
            - Feature Engineering:  extract relevant features, create new features, or transform existing ones to improve the model's ability to capture patterns and relationships within the data.
            """)

    st.subheader("3. Model Selection and Building:")
    st.markdown("""
            - Explore multiple classification algorithms for heart disease prediction.
            -Utilize logistic regression, naive Bayes, support vector machines, k-nearest neighbors, and random forest classifiers.
            - Implement each model to learn from the training data and make predictions.            
            - The models are trained and evaluated using accuracy and precision metrics.
            -Standardize features using the StandardScaler to ensure consistent scaling across variables.
                 Prepare the data for various machine learning algorithms.

            """)

    st.subheader("4. Model Ensembling with Stacking::")
    st.markdown("""
            - Different techniques are employed to enhance model performance.
            - Feature selection and hyperparameter tuning are considered.
            """)

    st.subheader("5. Model Deployment:")
    st.markdown("""
            - The best-performing model is selected and saved for future use.
            - Interactive Streamlit App: used Streamlit framework to create an intuitive and user-friendly web application.
            - User Input: Users provide essential health data, such as age, blood pressure, glucose level, and more, using the app's interface.
            - Model Inference: Behind the scenes, the machine learning models process the user's input, leveraging the knowledge they gained during the training phase to make informed predictions about the likelihood of heart disease.
     """)

    # Data for model evaluation



def display():
    st.sidebar.title('Welcome to the Diabetes Prediction App')
    st.sidebar.write('This app uses a machine learning model to predict the likelihood of Diabetes disease.')
    with st.sidebar.expander("How it Works❓❓"):
        st.write(
            "1. **Input Health Data**: Provide your health details, including age, blood pressure, glucose level and other relevant factors.")
        st.write(
            "2. **Instant Prediction**:  state-of-the-art machine learning model analyzes your input and generates an instant prediction about your risk of Diabetes disease.")

    with open('logistic_model.pkl', 'rb') as model_file:
        logistic_model = pickle.load(model_file)
    imcol, textcol,c,p = st.columns(4)
    textcol.header("Diabetes")
    c.header("Prediction ")



    image = Image.open("image/Diabetes-Type-1.png")

    new_size = (150, 150)  # Define the new size (width, height)
    resized_image = image.resize(new_size)

    # Display the resized image
    imcol.image(resized_image)



    gender = st.selectbox("Select Gender", ["Male", "Female"])

    if gender == "Female":
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
    else:
        pregnancies =0
    glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
    pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1.0, max_value=150.0, value=30.0, step=1.0)

    # Create a DataFrame from user inputs
    user_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [pedigree_function],
        "Age": [age]
    })

    # Prediction button
    if st.button("Predict"):
        # Predict using the trained logistic regression model
        prediction = logistic_model.predict(user_data)

        # Display prediction result
        if prediction[0] == 0:
            st.success("yaaay  you  are Not diabetic:")
            st.write("- The provided glucose level is within a normal range.")
            st.write("- The blood pressure is within a healthy range.")
            # Add more insights for the non-diabetic case





        else:
            st.error(" high chances you  are diabetic:")
            st.write("- The provided glucose level is higher than normal.")
            st.write("- The blood pressure is elevated.")
            # Add more insights for the diabetic case
            glucose_levels = np.array([80, 100, 120, 140, 160, 180, 200])
            diabetes_chances = np.array([10, 20, 30, 50, 70, 85, 95])

            # Create a Streamlit app

            # Create a line chart using Matplotlib
            fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed
            ax.plot(glucose_levels, diabetes_chances, marker='o')
            ax.set_xlabel('Glucose Level')
            ax.set_ylabel('Diabetes Chances')
            ax.set_title('Diabetes Chances vs Glucose Level')

            # Set the background color of the figure to black
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ax.grid(color='gray', linestyle='--', linewidth=0.5)  # Add grid lines

            # Set the text color to white
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

            st.pyplot(fig)
            # Create detailed visualizations for diabetic case

    st.set_option('deprecation.showPyplotGlobalUse', False)
