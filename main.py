import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#Tensorflow Model Prediction
def model_prediction(test_image):
    # Load our newly saved keras model
    model = tf.keras.models.load_model('crop_disease_detector_model.keras')
    # Convert uploaded image to PIL Image and resize to model's expected input (160, 160)
    image = Image.open(test_image).resize((160, 160))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home & About","Crop Disease Detector"])

#Merged Home & About Page
if(app_mode=="Home & About"):
    st.header("CROP DISEASE DETECTOR")
    st.markdown("""
    Welcome to the Crop Disease Detector!
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Crop Disease Detector** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### About the Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset consists of about 61486 images of healthy and diseased crop leaves which is categorized into 39 different classes.
    
    #### Dataset Content
    - **Train:** 49,179 images
    - **Validation:** 6,139 images
    - **Test:** 6,168 images

    ### Get Started
    Click on the **Crop Disease Detector** page in the sidebar to upload an image and experience the power of our Crop Disease Detector!
""")

#Prediction Page
elif(app_mode=="Crop Disease Detector"):
    st.header("Crop Disease Detector")
    test_image = st.file_uploader("Choose an Image:")
    if(test_image is not None and st.button("Show Image")):
        st.image(test_image,width="stretch")
    #Predict Button
    if(test_image is not None and st.button("Predict")):
        with st.spinner("Please Wait.."):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            # Define Classes based on the dataset's class_names
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
                'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
                'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))