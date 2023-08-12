import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import  numpy as np
from PIL import  Image


def show_project_overview_page():
    st.header("Introduction to the Kidney Disease Prediction Project")

    st.markdown("""
    Understand your kidney disease risk using advanced machine learning. Input your health data to receive personalized predictions and insights. With 98% accurate results, take charge of your health.
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
    - Explore multiple classification algorithms for kidney disease prediction.
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
    - User Input: Users provide essential health data, such as age, blood pressure, glucose level, and more, using the app's interface.
    - Model Inference: Behind the scenes, the machine learning models process the user's input, leveraging the knowledge they gained during the training phase to make informed predictions about the likelihood of kidney disease.
    """)

    # Data for model evaluation



def display():
    st.sidebar.title('Welcome to the Kidney Disease Prediction App')
    st.sidebar.write('This app uses a machine learning model to predict the likelihood of Kidney disease.')
    with st.sidebar.expander("How it Works❓❓"):
        st.write(
            "1. **Input Health Data**: Provide your health details, including Diabetes Mellitus,Hypertension,Haemoglobin and other relevant factors.")
        st.write(
            "2. **Instant Prediction**:  state-of-the-art machine learning model analyzes your input and generates an instant prediction about your risk of Kidney disease.")


    heatmap_df = pd.read_csv('kidneyHeatMap.csv')
    model_filename = 'models/etckidney.pkl'
    with open(model_filename, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    imcol, textcol  = st.columns(2)

    textcol.header("Kidney Disease Prediction")

    image = Image.open("image/kidney.png")

    new_size = (150, 150)  # Define the new size (width, height)
    resized_image = image.resize(new_size)

    # Display the resized image
    imcol.image(resized_image)




    # Display the styled pop-up using HTML and CSS

    specific_gravity = st.slider("Specific Gravity", 1.005, 1.025, step=0.001, format="%.3f",help="Measures urine concentration, reflecting kidney waste-filtering ability.  try:1.020")

    diabetes_mellitus_map = {'yes': 1, 'no': 0}
    diabetes_mellitus = st.selectbox("Diabetes Mellitus", list(diabetes_mellitus_map.keys()),help = " High blood sugar; harms kidneys,    try: yes")
    diabetes_mellitus = diabetes_mellitus_map[diabetes_mellitus]

    hypertension_map = {'yes': 1, 'no': 0}
    hypertension = st.selectbox("Hypertension", list(hypertension_map.keys()),help = "High blood pressure strains kidneys, affects waste removal., try: yes")
    hypertension = hypertension_map[hypertension]

    albumin_values = [0.0, 2.0, 3.0, 1.0, 4.0, 5.0]
    albumin = st.selectbox("Albumin", albumin_values,help = "Protein in urine signals kidney function issues., try: 1.0")
    packed_cell_volume = st.number_input("Packed Cell Volume", min_value=15.0, max_value=50.0,value=30.0 ,help = "Shows red blood cell percentage, indicating anemia risk, try: 44.0")
    haemoglobin = st.number_input("Haemoglobin",  max_value=20.0, value=14.0,help = "Carries oxygen, low levels cause fatigue due to kidney issues., try:15.4")

    pus_cell_map = {'normal': 1, 'abnormal': 0}
    pus_cell = st.selectbox("Pus Cell", list(pus_cell_map.keys()),help = "Urinary infection indicator, more concerning in kidney disease.,   try: normal")
    pus_cell = pus_cell_map[pus_cell]

    appetite_map = {'good': 0, 'poor': 1}
    appetite = st.selectbox("Appetite", list(appetite_map.keys()),help = ": Changes occur in kidney disease; vital for overall health and kidney support. try: good")
    appetite = appetite_map[appetite]
    # Prepare user input for predictio

    # Make prediction when Predict button is pressed

    predict_button = st.button("Predict")

    if predict_button:
        # Prepare user input for prediction
        user_input = pd.DataFrame({
            'specific_gravity': [specific_gravity],
            'diabetes_mellitus': [diabetes_mellitus],
            'hypertension': [hypertension],
            'albumin': [albumin],
            'packed_cell_volume': [packed_cell_volume],
            'haemoglobin': [haemoglobin],
            'pus_cell': [pus_cell],
            'appetite': [appetite]
        })

        # Make prediction using the loaded model
        prediction = loaded_model.predict(user_input)

        # Display the prediction result
        st.write(f"Prediction: {prediction[0]}")
        if prediction[0] == 0:
            st.error("You are a patient of chronic kidney disease:")
            st.write("- The provided features indicate a potential kidney disease.")
            st.write("- Specific Gravity: Indicates a certain level of kidney function.")
            st.write("- Albumin: Elevated levels a possible  sign of kidney damage.")
            feature_importances = {
                'Specific Gravity': 0.144211,
                'Diabetes Mellitus': 0.141934,
                'Hypertension': 0.121480,
                'Albumin': 0.114679,
                'Packed Cell Volume': 0.083572,
                'Haemoglobin': 0.059435,
                'Pus Cell': 0.048770,
                'Appetite': 0.045480
            }
            plt.style.use('dark_background')

            # Create a bar plot to visualize feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(list(feature_importances.keys()), list(feature_importances.values()))
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.title('Factors Affecting Kidney Disease Prediction')
            st.pyplot(plt)
            # Display SHAP values for each feature
            st.write("Correlation Heatmap:")
            correlation_matrix = heatmap_df.corr()
            sns.set(style="white")
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            cmap = sns.color_palette("coolwarm", as_cmap=True)  # Choose a color map
            plt.figure(figsize=(10, 8))
            cbar_kws = {"ticks": [-1, -0.5, 0, 0.5, 1], }

            ax = sns.heatmap(heatmap_df.corr(), annot=True, linewidths=1, linecolor='lightgrey')
            # Change label text color
            for label in ax.get_xticklabels():
                label.set_color('white')  # Change x-axis label text color

            for label in ax.get_yticklabels():
                label.set_color('white')  # Change y-axis label text color
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelcolor='white')
            st.pyplot(plt)
            # Add more insights related to other features
        else:
            st.success("You are not a patient of chronic kidney disease:")
            st.write("- The provided features suggest a healthier condition.")
            st.write("- Haemoglobin: Within a normal range, indicating overall health.")
            # Add more insights related to other features
            feature_importances = {
                'Specific Gravity': 0.144211,
                'Diabetes Mellitus': 0.141934,
                'Hypertension': 0.121480,
                'Albumin': 0.114679,
                'Packed Cell Volume': 0.083572,
                'Haemoglobin': 0.059435,
                'Pus Cell': 0.048770,
                'Appetite': 0.045480
            }
            plt.style.use('dark_background')

            # Create a bar plot to visualize feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(list(feature_importances.keys()), list(feature_importances.values()))
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.title('Factors Affecting Kidney Disease Prediction')
            st.pyplot(plt)


            st.write("Correlation Heatmap:")
            correlation_matrix = heatmap_df.corr()
            sns.set(style="white")
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            cmap = sns.color_palette("coolwarm", as_cmap=True)  # Choose a color map
            plt.figure(figsize=(10, 8))
            cbar_kws = {"ticks": [-1, -0.5, 0, 0.5, 1], }

            ax = sns.heatmap(heatmap_df.corr(),annot=True, linewidths=1, linecolor='lightgrey')
            # Change label text color
            for label in ax.get_xticklabels():
                label.set_color('white')  # Change x-axis label text color

            for label in ax.get_yticklabels():
                label.set_color('white')  # Change y-axis label text color
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelcolor='white')
            st.pyplot(plt)

