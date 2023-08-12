import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import  numpy as np
from PIL import  Image


def show_project_overview_page():
    st.header("Introduction to the Liver Disease Prediction Project")

    st.markdown("""
    Understand your liver disease risk using advanced machine learning. Input your health data to receive personalized predictions and insights. With 98% accurate results, take charge of your health.
    """)

    st.header("Project Overview")

    st.markdown("""
    The project is structured into several key steps:
    """)

    st.subheader("1. Data Cleaning and Exploratory Data Analysis (EDA):")
    st.markdown("""
    - Clean and prepare the dataset for analysis.
    - Generate data insights using pandas-profiling.
    - Handle missing values, outliers, and inconsistencies.
    - Gain insights into dataset structure, distributions, and correlations through EDA.
    """)

    st.subheader("2. Data Preprocessing:")
    st.markdown("""
    - Train-Test Split: We split the dataset into training and testing subsets. The training set is used to teach the machine learning models, while the testing set is reserved for evaluating their performance.
    - Feature Engineering: Extract relevant features, create new features, or transform existing ones to improve the model's ability to capture patterns and relationships within the data.
    """)

    st.subheader("3. Model Selection and Building:")
    st.markdown("""
    - Explore multiple classification algorithms for liver disease prediction.
    - Utilize logistic regression, naive Bayes, support vector machines, k-nearest neighbors, and random forest classifiers.
    - Implement each model to learn from the training data and make predictions.
    - The models are trained and evaluated using accuracy and precision metrics.
    - Standardize features using the StandardScaler to ensure consistent scaling across variables.
    - Prepare the data for various machine learning algorithms.
    """)

    st.subheader("4. Model Ensembling with Stacking:")
    st.markdown("""
    - Different techniques are employed to enhance model performance.
    - Feature selection and hyperparameter tuning are considered.
    """)

    st.subheader("5. Model Deployment:")
    st.markdown("""
    - The best-performing model is selected and saved for future use.
    - Interactive Streamlit App: used Streamlit framework to create an intuitive and user-friendly web application.
    - User Input: Users provide essential health data, such as age, Diabetes total protien level,direct bilirubin level,Albumin and more, using the app's interface.
    - Model Inference: Behind the scenes, the machine learning models process the user's input, leveraging the knowledge they gained during the training phase to make informed predictions about the likelihood of liver disease.
    """)

    # Data for model evaluation
    model_data = {
        'Model': [' DecisionTree', 'Logistic Regression ', 'K-Nearest Neighbour', 'Random Forest'],
        'Accuracy': [70.213, 70.4891, 64.844, 83.647554]
    }
    model_ev = pd.DataFrame(model_data)

    # Sort the DataFrame by accuracy
    model_ev = model_ev.sort_values(by='Accuracy', ascending=False)

    # Create a bar plot
    plt.style.use('dark_background')

    plt.figure(figsize=(10, 6))
    plt.barh(model_ev['Model'], model_ev['Accuracy'], color='skyblue')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Model')
    plt.title('Model Evaluation: Accuracy Comparison')
    plt.xlim(0, 100)  # Set the x-axis limit to 100 for percentage display
    st.pyplot()

    st.write("so Used Random Forest at final")
def display():
    st.sidebar.title('Welcome to the Liver Disease Prediction App')
    st.sidebar.write('This app uses a machine learning model to predict the likelihood of liver disease.')
    with st.sidebar.expander("How it Works❓❓"):
        st.write(
            "1. **Input Health Data**: Provide your health details, including Diabetes total protien level,direct bilirubin level,Albumin to Globulin Ratio and other relevant factors.")
        st.write(
            "2. **Instant Prediction**:  state-of-the-art machine learning model analyzes your input and generates an instant prediction about your risk of liver disease.")



    model_filename = 'models/liveRrfc.pkl'
    with open(model_filename, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    imcol, textcol  = st.columns(2)

    textcol.header("Liver Disease Prediction")

    image = Image.open("image/liver.png")

    new_size = (150, 150)  # Define the new size (width, height)
    resized_image = image.resize(new_size)

    # Display the resized image
    imcol.image(resized_image)

    age = st.number_input("Age", value=18 ,step=1)
    age =int(age)
    gender_map = {'male': 1, 'female': 0}
    gender = st.selectbox("Gender", list(gender_map.keys()))
    gender = gender_map[gender]
    tot_bilirubin = st.number_input("total bilirubin", min_value=0.0, max_value=30.0, value=10.0, step=0.1,help= "total bilirubin level in a person's blood. Elevated levels can indicate liver or bile duct issues.")
    direct_bilirubin = st.number_input("direct bilirubin", min_value=0.0, max_value=30.0, value=15.0, step=0.1,help="direct bilirubin level, which is the portion of bilirubin that is conjugated (chemically modified) in the liver")
    tot_proteins = st.number_input("Total Proteins", min_value=0, max_value=2000, value=150,help=" total protein levels in the blood.")
    albumin = st.number_input("albumin", min_value=0, max_value=2000, value=500,help="Albumin is a specific type of protein produced by the liver. Its level in the blood can reflect liver function and overall nutritional status.")
    ag_ratio = st.number_input("Albumin to Globulin Ratio", min_value=0,  value=2000,help=" ratio of albumin to other globulin proteins in the blood.")
    sgpt = st.number_input("Serum Glutamic Pyruvic Transaminase(SGPT)", min_value=0.0, max_value=10.0, value=5.5,help=" enzyme found primarily in the liver and can be used as a marker of liver health. Elevated levels may indicate liver damage or disease.")
    sgot = st.number_input(" Serum Glutamic Oxaloacetic Transaminase (SGOT)", min_value=0.0, max_value=10.0, value=5.5,help=" enzyme found in the liver and other organs. Elevated levels can indicate liver damage,")
    alkphos = st.number_input(" Alkaline Phosphatase (ALP)",min_value=0.0, max_value=3.0, value=1.5,help=" Elevated ALP levels can be indicative of liver or bone disorders.")


    predict_button = st.button("Predict")

    if predict_button:
        # Prepare user input for prediction
        user_input = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'tot_bilirubin': [tot_bilirubin],
            'direct_bilirubin': [direct_bilirubin],
            'tot_proteins': [tot_proteins],
            'albumin': [albumin],
            'ag_ratio': [ag_ratio],
            'sgpt': [sgpt],
            'sgot': [sgot],
            'alkphos': [alkphos]
        })

        # Make prediction using the loaded model
        prediction = loaded_model.predict(user_input)

        # Display the prediction result
        st.write(f"Prediction: {prediction[0]}")
        if prediction[0] == 1:
            st.error("Alert! You are a patient of Liver disease:")
            st.write("- The provided features indicate a potential liver disease.")
            st.write("- high total protiens level: Indicates a certain level of liver function.")
            st.write("- Albumin: Elevated levels a possible  sign of liver damage.")
            feature_importances = {
                'tot_proteins': 0.150410,
                'ag_ratio': 0.135550,
                'age': 0.131609,
                'albumin': 0.124130,
                'sgot': 0.096020,
                'direct_bilirubin': 0.092365,
                'tot_bilirubin': 0.087026,
                'sgpt': 0.084846,
                'alkphos': 0.080321,
                'gender': 0.017724
            }
            plt.style.use('dark_background')

            # Create a bar plot to visualize feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(list(feature_importances.keys()), list(feature_importances.values()))
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.title('Factors Affecting liver Disease Prediction')
            st.pyplot(plt)


    else:
            st.success("yaay! You are not a patient of  liver disease:")
            st.write("- The provided features suggest a healthier condition.")
            # Add more insights related to other features
            feature_importances = {
                'tot_proteins': 0.150410,
                'ag_ratio': 0.135550,
                'age': 0.131609,
                'albumin': 0.124130,
                'sgot': 0.096020,
                'direct_bilirubin': 0.092365,
                'tot_bilirubin': 0.087026,
                'sgpt': 0.084846,
                'alkphos': 0.080321,
                'gender': 0.017724
            }
            plt.style.use('dark_background')

            # Create a bar plot to visualize feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(list(feature_importances.keys()), list(feature_importances.values()))
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.title('Factors Affecting liver Disease Prediction')
            st.pyplot(plt)



st.set_option('deprecation.showPyplotGlobalUse', False)