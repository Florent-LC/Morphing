from facenet_pytorch import MTCNN
import dlib
import cv2
from code.Image_manipulation import Image
import streamlit as st



mtcnn = MTCNN(keep_all=True)
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



class WebCamVideo :

    def __init__(self) :

        self.video_capture = cv2.VideoCapture(0)
        
        self.mtcnn = MTCNN(keep_all=False)        
        self.dlib = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    
    def detect_face_live(self, image1 : st.delta_generator.DeltaGenerator, detect : bool) :
        
        while(self.video_capture.isOpened()):

            retrieve, frame = self.video_capture.read()

            if not(retrieve) :
                st.write("Frame not retrieve")
                break

            img = Image(frame, mtcnn = self.mtcnn, dlib = self.dlib, model = False, title="Face_detection")
            if detect :
                img.detect_faces()
            
            image = img.img
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image1.image(image,use_column_width=True,caption="Transformed input")


        self.video_capture.release()
        cv2.destroyAllWindows()
        

    def morphing_realtime(self, image1 : st.delta_generator.DeltaGenerator, img_model : Image) :

        while(self.video_capture.isOpened()):

            retrieve, frame = self.video_capture.read()

            if not(retrieve) :
                st.write("Frame not retrieve")
                break

            img_applied = Image(frame, mtcnn = self.mtcnn, dlib = self.dlib, model = False)
            img_morphing = Image.morphing(img_model,img_applied)

            if img_morphing is None:
                img_morphing = img_applied.img

            cv2.imshow("Morphing Image", img_morphing)

            img_morphing = cv2.cvtColor(img_morphing, cv2.COLOR_BGR2RGB)
            
            image1.image(img_morphing,use_column_width=True,caption="Transformed input")

        self.video_capture.release()
        cv2.destroyAllWindows()