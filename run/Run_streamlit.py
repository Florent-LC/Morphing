r"""
In order to run this file, use this command line, and make sure to be on the parent directory (.\Morphing):

python -m streamlit run .\run\Run_streamlit.py

"""


import cv2
import streamlit as st
from code.Webcam_streamlit import WebCamVideo
from code.Image_manipulation import Image
import numpy as np



def main():

    img_model = Image.get_face(title="Generated Face", model=True)

    st.title('Face Morphing or Detection')

    col1, col2 = st.columns(2)

    img_model_image = img_model.img
    img_model_image = cv2.cvtColor(img_model_image, cv2.COLOR_BGR2RGB)

    image1 = col1.image(np.zeros_like(img_model_image),use_column_width=True, caption="Transformed input")
    image2 = col2.image(img_model_image,use_column_width=True, caption="Generated Image")

    change_face_button = col2.button("Change face", key="change_face_button")
    if change_face_button :
        image2.image(img_model_image,use_column_width=True, caption="Generated Image")

    stop_animation_button = col1.checkbox("Stop", key="Stop")

    action_selection = st.sidebar.selectbox('What do you want to do?',options=["Nothing", "Detection", "Morphing"])

    video = WebCamVideo()

    if not(stop_animation_button) :
        if action_selection == "Nothing":
            video.detect_face_live(detect=False, image1=image1)
        elif action_selection == "Detection":
            video.detect_face_live(detect=True, image1=image1)
        else:
            video.morphing_realtime(image1=image1, img_model=img_model)

    

if __name__ == '__main__':
    main()
