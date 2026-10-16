import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#Tensorflow Model Prediction
def model_prediction(test_image):
    # Load our newly saved keras model
    model = tf.keras.models.load_model('crop_disease_detector_model.keras')
    # Convert uploaded image to PIL Image,convert to RGB and resize to model's expected input (160, 160)
    image = Image.open(test_image).convert("RGB").resize((160, 160))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","Crop Disease Detector"])

#Home Page
if(app_mode=="Home"):
    st.header("CROP DISEASE DETECTOR")
    st.markdown("""
    Welcome Farmer!
    
    Our mission is to help you identify any crop diseases on your farm. 
    Together, let's protect your crops and ensure you have a bountiful harvest!

    ### Here is how our Detector Works
    1. **Upload your image** on the **Crop Disease Detector** page and click on "Diagnose"
    2. **Wait for our Detector to analyze your image**...Takes a few seconds :)
    3. **View the results and take the necessary actions**.

    ### Why use our Detector?
    - **It is user-friendly**
    - **It is fast and accurate** 

    ### About
    The dataset used was curated from Mendeley and it consists of **61486 images** of healthy and diseased crop leaves which is categorized into 39 different classes.
    Our Model was trained with **49,179 images**, tested with **6,168 images** and validated with **6,139 images** with up to **98% accuracy** achieved. The project was handled by 4 techies from Techcrush 
    Bootcamp AL/ML Track: Nnaemeka, Tega, Molly and Angel.
    

    ### Head over to the **Crop Disease Detector** page in the sidebar to get started!
""")

# Detector Page
elif(app_mode=="Crop Disease Detector"):
    st.header("Crop Disease Detector")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if(test_image is not None and st.button("Show Uploaded Image")):
        st.image(test_image,width="stretch")
    #Predict Button
    if(test_image is not None and st.button("Diagnose")):
        with st.spinner("Please Wait for the Results.."):
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
