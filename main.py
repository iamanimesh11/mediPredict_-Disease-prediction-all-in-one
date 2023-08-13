import streamlit as st
import heartDisease
import diabetes
import  kidney
import  liver
from PIL import Image

st.set_page_config(page_title='MediPredict ',page_icon="🤖")

# Add JavaScript to hide the GitHub icon
hide_github_icon_js = """
<style>
#MainMenu {
    display: none;
}
button.css-ch5dnh {
    display: none;
}
</style>
<script>
document.addEventListener('DOMContentLoaded', function() {
    const toolbar = document.querySelector('[data-testid="stToolbar"]');
    if (toolbar) {
        toolbar.style.display = 'none';
    }
});
</script>
"""
st.markdown(hide_github_icon_js, unsafe_allow_html=True)

def medical():
    st.sidebar.write("Discover insights into your well-being through advanced machine learning models that offer Predictions of heart dieesease,liver disease,kidney disease ,diabete disease. User friendly interface helps you to input the details and let the models in backend works upon it .Accuray are relative high of all prediction model .Don't just get prediction also visualise the details as  per required and know what factors are major impact to that particular disease .Try now :)")

    imcol,textcol,x = st.columns(3)
    image = Image.open("image/Health-PNG-Pic.png")

    new_size = (200, 200)  # Define the new size (width, height)
    resized_image = image.resize(new_size)
    # Display the resized image
    imcol.image(resized_image)
    # x.image(resized_image)
    st.header("Multi-Diseases  Predictions AI")
    textcol.title("MediPredict")
    textcol.markdown(" &#160; &#160; &#160;Made by Animesh | ")
    textcol.write("&#160; &#160; &#160;[website](https://share.streamlit.io/app/animesh11portfolio/) | [LinkedIn](https://www.linkedin.com/in/animesh-singh11)")

    data = {
        "Heart Disease Prediction": "image/heart2.png",

        "Liver Disease Prediction": "image/liver.png",
        "Kidney Disease Prediction":"image/kidney.png",
        "Diabetes Prediction": "image/Diabetes-Type-1.png",

        # "Mental Health Prediction": "mental.png"
    }
    for disease_name, image_filename in data.items():
        img = Image.open(image_filename)
        img = img.resize((150, 150))  # Resize images to a common size
        new_filename = f"resized_{image_filename}"  # Save the resized images with a new name
        img.save(new_filename)

    row = st.columns(5)
    row = st.container()
    for i in range(0, len(data), 3):
        row = st.columns(3)
        for j in range(3):
            if i + j < len(data):
                with row[j]:
                    disease_name = list(data.keys())[i + j]
                    if st.button(disease_name):
                        st.success("Check Sidebar :)")

                    image_filename = data[disease_name]
                    st.image(f"resized_{image_filename}", width=150)

    st.sidebar.success("check my others projects too :)")
    st.sidebar.markdown(
        "🍿||  [cinema_nexus ](https://cinemanexus.streamlit.app/) ||🍿 ")
    st.sidebar.markdown(
        "💬||[spambuster_ai ∙](https://spambusterai.streamlit.app/)||💬")
    st.sidebar.markdown(
        "📚||  [intelligence_books_suggester_app ∙](https://intelligencebookssuggesterapp.streamlit.app/)||📚")
    st.sidebar.markdown(
        "▶️||  [youtube_video_sentimentsAnalysis..app∙](https://youtubevideosentimentanalysis.streamlit.app/)  ||▶️")


def main():

    navigation = st.sidebar.selectbox(
        "",
        ("MediPredict" , "Heart Disease Prediction", "Diabetes Prediction","Kidney Disease Prediction","Liver Disease Prediction")
    )

    if navigation == "MediPredict":
        medical()

    elif navigation == "Heart Disease Prediction":
        sub_page = st.sidebar.radio("", ["Heart Disease prediction", "project overview"])
        if sub_page == "Heart Disease prediction":
            heartDisease.display()


        elif sub_page == "project overview":
            heartDisease.show_project_overview_page()

    elif navigation == "Diabetes Prediction":
        sub_page = st.sidebar.radio("", ["Diabetes Prediction", "project overview"])
        if sub_page == "Diabetes Prediction":
            diabetes.display()


        elif sub_page == "project overview":
            diabetes.show_project_overview_page()
    elif navigation == "Kidney Disease Prediction":
        sub_page = st.sidebar.radio("", ["Kidney Disease Prediction", "project overview"])
        if sub_page == "Kidney Disease Prediction":
            kidney.display()
        elif sub_page == "project overview":
            kidney.show_project_overview_page()

    elif navigation == "Liver Disease Prediction":
        sub_page = st.sidebar.radio("", ["Liver Disease Prediction", "project overview"])
        if sub_page == "Liver Disease Prediction":
            liver.display()
        elif sub_page == "project overview":
            liver.show_project_overview_page()


if __name__ == '__main__':
    main()
