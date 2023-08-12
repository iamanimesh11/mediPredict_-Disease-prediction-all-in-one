import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def show_project_overview_page():
    st.header("Introduction to the Heart Disease Prediction Project")

    st.markdown("""
    In the "Heart Disease Prediction" section, you can assess your risk of heart disease using advanced machine learning technology. This feature allows you to input key health information and receive valuable insights about your heart health.
                """)

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
            -Utilize logistic regression, naive Bayes, decision trees, support vector machines, extreme gradient boosting, k-nearest neighbors, and random forest classifiers.
            - Implement each model to learn from the training data and make predictions.            
            - The models are trained and evaluated using accuracy and precision metrics.
            -Standardize features using the StandardScaler to ensure consistent scaling across variables.
                 Prepare the data for various machine learning algorithms.
            
            """)

    st.subheader("4. Model Ensembling with Stacking::")
    st.markdown("""
            - Different techniques are employed to enhance model performance.
            - Feature selection and hyperparameter tuning are considered.
            - Employ the StackingCVClassifier from mlxtend to create a powerful ensemble model.
            - Combine the strengths of various classifiers for improved prediction accuracy.
            """)

    st.subheader("5. Model Deployment:")
    st.markdown("""
            - The best-performing model is selected and saved for future use.
            - Interactive Streamlit App: used Streamlit framework to create an intuitive and user-friendly web application.
            - User Input: Users provide essential health data, such as age, blood pressure, cholesterol levels, and more, using the app's interface.
            - Model Inference: Behind the scenes, the machine learning models process the user's input, leveraging the knowledge they gained during the training phase to make informed predictions about the likelihood of heart disease.
     """)

    # Data for model evaluation
    model_data = {
        'Model': ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'Extreme Gradient Boost',
                  'K-Nearest Neighbour', 'Decision Tree', 'Support Vector Machine'],
        'Accuracy': [86.341463, 85.365854, 93.658537, 94.634146, 87.804878, 94.634146, 98.048780]
    }

    # Create a DataFrame
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


def display():

    # Load the trained model
    st.sidebar.title('Welcome to the Heart Disease Prediction App')
    st.sidebar.write('This app uses a machine learning model to predict the likelihood of heart disease.')
    with st.sidebar.expander("How it Works❓❓"):
        st.write("1. **Input Health Data**: Provide your health details, including age, blood pressure, cholesterol levels, and other relevant factors.")
        st.write("2. **Instant Prediction**: Our state-of-the-art machine learning model analyzes your input and generates an instant prediction about your risk of heart disease.")
        st.write("3. **Actionable Insights**: Based on the prediction, displays chances of heart disease")
    model_filename = 'models/heartsvg.pkl'
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)

    # Streamlit app title

    imcol,textcol= st.columns(2)
    textcol.title("Heart Disease Prediction ")

    image = Image.open("image/heart.png")

    new_size = (150, 150)  # Define the new size (width, height)
    resized_image = image.resize(new_size)

    # Display the resized image
    imcol.image(resized_image)
    def visualize_max_heart_rate():
        # Simulated data for demonstration
        max_heart_rates_high_chance = np.random.randint(70, 100, 100)
        max_heart_rates_low_chance = np.random.randint(100, 180, 100)

        # Set the background color of the plot
        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')  # Set the dark background style

        sns.histplot(max_heart_rates_high_chance, color='red', label='High Chance')
        sns.histplot(max_heart_rates_low_chance, color='blue', label='Low Chance')

        # Customize other plot properties (title, labels, etc.)
        plt.xlabel('Maximum Heart Rate Achieved', color='white')  # Set the color of the x-axis label
        plt.ylabel('Frequency', color='white')  # Set the color of the y-axis label
        plt.title('Distribution of Maximum Heart Rates for Different Predictions')
        plt.legend()

        # Set the color of the legend text
        legend = plt.legend()
        for text in legend.get_texts():
            text.set_color('white')

        # Set the color of the tick labels on the axes
        plt.xticks(color='white')
        plt.yticks(color='white')

        st.pyplot()


    age = st.number_input("Age:")
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg):",help="try:125")
    chol = st.number_input("Serum Cholesterol (mg/dL):",help="try:212")
    fbs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dL", "> 120 mg/dL"])
    restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved:",help="try:168")
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression Induced by Exercise:",help="try:1")
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy:", min_value=0, max_value=3)
    thal = st.selectbox("Thallium Stress Test Result", ["Normal", "Fixed Defect", "Reversible Defect"])


    cp_encoded = LabelEncoder().fit_transform([cp])[0]
    restecg_encoded = LabelEncoder().fit_transform([restecg])[0]
    slope_encoded = LabelEncoder().fit_transform([slope])[0]
    thal_encoded = LabelEncoder().fit_transform([thal])[0]

    # Convert the input features to a 2D array
    predict_button = st.button("Predict Heart Disease")
    if predict_button:
        # Convert the input features to a 2D array
        sex_encoded = 1 if sex == "Male" else 0

        fbs_encoded = 1 if fbs == "> 120 mg/dL" else 0
        exang_encoded = 1 if exang == "Yes" else 0

        user_input = np.array([[age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded, restecg_encoded, thalach,
                                exang_encoded, oldpeak, slope_encoded, ca, thal_encoded]])

        # Make a prediction based on user input
        prediction = loaded_model.predict(user_input)

        # Display the prediction and insights
        if prediction[0] == 1:
            st.error("Prediction: High Chance of Heart Disease")
            st.write("Factors contributing to high chance:")
            if sex_encoded == 1:
                st.write("- Male sex")

            else:
                st.write("- Female sex")

            if cp_encoded == 0:
                st.write("- Typical Angina chest pain")

            if thalach < 100:
                st.write(
                    f"- Lower Maximum Heart Rate ({thalach}): A lower maximum heart rate achieved during exercise may suggest reduced cardiovascular fitness.")

            if exang_encoded == 1:
                st.write(
                    "- Exercise Induced Angina: Experiencing angina during exercise can indicate an increased likelihood of heart disease.")
            # Add more explanations for other features
            visualize_max_heart_rate()

        else:
            st.success("Prediction: Low Chance of Heart Disease")
            st.write("Factors contributing to low chance:")
            if sex_encoded == 0:
                st.write("- Female sex")
            if cp_encoded == 2:
                st.write(
                    "- Non-Anginal Pain chest pain: This type of chest pain is less likely to be associated with heart issues.")
            if thalach > 150:
                st.write(
                    f"- Higher Maximum Heart Rate ({thalach}): A higher maximum heart rate achieved during exercise suggests better cardiovascular fitness.")
            visualize_max_heart_rate()

    st.set_option('deprecation.showPyplotGlobalUse', False)


