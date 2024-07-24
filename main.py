import streamlit as st
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import os

st.set_page_config(
    page_title="Crater Detection",
    page_icon="🔭",
    layout="centered"
)

# Custom CSS to set the background image and font colors
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        bg_image = image_file.read()
        b64_image = base64.b64encode(bg_image).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{b64_image}");
            background-size: cover;
            color: white;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: white;
        }}
        .css-1d391kg {{
            color: black;
        }}
        /* Apply to the sidebar and its elements */
        .css-1d391kg {{
            color: black; /* Sidebar title */
        }}
        .css-1d391kg .css-1v3fvcr {{
            color: black; /* Sidebar text */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Your main code here
add_bg_from_local('static/bg-image.jpeg')

# TensorFlow Model Prediction
def model_prediction(test_image):
    model_path = 'lunar-crater.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    image = Image.open(test_image)
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Add batch dimension
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Lunar Crater Detection"])

if app_mode == "Home":
    st.header("Lunar Crater Detection System")
    st.markdown("""
    Welcome to our Lunar Crater Detection System! 🌕🔭

    Our mission is to help you identify lunar craters efficiently. Please upload an image of the lunar surface, and our system will analyze it to detect any signs of craters. Together, let's explore the moon and enhance our understanding of its topography!

    ### How It Works
    1. **Upload Image:** Go to the **Crater Detection** page and upload an image of the lunar surface.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential craters.
    3. **Results:** View the results and recommendations for further analysis.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate crater detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Crater Detection** page in the sidebar to upload an image and experience the power of our Lunar Crater Detection System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ## About Dataset
    ### Why this dataset
    Efficient detection of craters can be of vital significance in various space exploration missions. Previous researches have already made significant progress on this task, however the versatility and robustness of existing methods are still limited. While modern object detection methods using deep learning is gaining popularity and is probably a solution to aforementioned problems, public-accessible data for training is hard to find. This is the primary reason we propose this dataset.

    ### What's in the dataset
    The dataset mainly contains:

    Image Data: Images of Mars and Moon surface which MAY contain craters. The data source is mixed. For Mars images, images are mainly from ASU and USGS; Currently all Moon images are from NASA Lunar Reconnaissance Orbiter mission. All images are preprocessed with RoboFlow to remove EXIF rotation and resize to 640*640.
    Labels: Each image has its associated labelling file in YOLOv5 text format. The anotation work was performed by ourselves, and mainly serves the purpose of object detection.
    Trained YOLOv5 model file: For each new version, we will upload our pretrained YOLOv5 model file using the latest version of data. The network strcture currently in use is YOLOv5m6.
    
    #### Link to Dataset
    Link: https://www.kaggle.com/datasets/lincolnzh/martianlunar-crater-detection-dataset
    """)

elif app_mode == "Lunar Crater Detection":
    st.header("Lunar Crater Recognition🌑")
    test_image = st.file_uploader("Upload an Image:")
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
        if st.button("Predict"):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            if result_index is not None:
                class_name = ['Crater', 'Not Crater']
                st.success(f"According to our Model, it is a {class_name[result_index]}.")
