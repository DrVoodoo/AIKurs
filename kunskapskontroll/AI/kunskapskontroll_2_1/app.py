import numpy as np
import streamlit as st
from PIL import Image

# third party components
from streamlit_drawable_canvas import st_canvas

import joblib

# load the model that was saved from the jupyter notebook file create_mnist_model_for_streamlit.ipynb
clf_extra_trees = joblib.load("mnist_model.pkl")

nav = st.sidebar.radio("Navigation Menu",["Purpose", "Predict Number"])

if nav == "Purpose":
    st.title("Streamlit - AI course kunskapskontroll 2 part 1")
    st.header("Purpose")
    st.write("""The purpose of this is get an overview of streamlit and create an app that can predict numbers from drawings. 
             An Extra trees model has been choosen to do the predictions. The MNIST dataset has been used for training the model.
             The third party component streamlit-drawable-canvas has been used to speed up the development.
             From the beginning I had an idea to try the WebRTC third party component but took the faster route with just drawing :).""")
    
if nav == "Predict Number":
    st.title("Classify a number")
    st.write("Draw a number and see what the extra tree classifier will interpret it as")

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#fff")

    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    canvas = st_canvas(
        fill_color="white",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        width=140,
        height=140,
        update_streamlit=realtime_update,
        drawing_mode='freedraw'
    )
    
    if canvas.image_data is not None:
        img = Image.fromarray(canvas.image_data.astype(np.uint8))
        
        # Convert to grayscale (MNIST is grayscale)
        img = img.convert("L")

        # Resize to 28x28 pixels (standard size for MNIST)
        img = img.resize((28, 28))

        # Convert the image to a NumPy array
        img_array = np.array(img)

        # Normalize the image (MNIST is in the range of 0 to 255, and sometimes we want to scale it)
        img_array = 255 - img_array

        # only do this if something has been drawn
        if np.sum(img_array) > 0:
            # Flatten the image to match the MNIST format (28x28 -> 784)
            img_array = img_array.flatten()

            st.write(f"Preprocessed image:")
            st.image(img, width=140)
            
            adjusted_clf_value = []
            adjusted_clf_value.append(img_array)
            predicted_number = clf_extra_trees.predict(adjusted_clf_value)
            
            st.write("Number predicted from the image:")
            st.write(f"# {predicted_number[0]}")
