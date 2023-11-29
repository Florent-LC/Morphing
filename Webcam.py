import cv2
from Image import *
 

class WebCamVideo :

    def __init__(self, fps : float = 30., exit_keybind : str = "\x1b") :

        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.fps = fps

        # the delay between two frames in ms
        self.delay = int(1000 / self.fps)
        self.exit_keybind = exit_keybind # "Esc" keybind
        
        self.mtcnn = MTCNN(keep_all=False)        
        self.dlib = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    
    def detect_face_live(self) :

        if not(self.video_capture.isOpened()) :
            raise Exception("An error occured trying to access the WebCam")

        while(self.video_capture.isOpened()):

            retrieve, frame = self.video_capture.read()

            if not(retrieve) :
                raise Exception("An error occured during the recording")

            img = Image(frame, mtcnn = self.mtcnn, dlib = self.dlib, model = False, title="Face_detection")
            img.detect_faces()
            img.show()

            key = cv2.waitKey(self.delay)
     
            if key == ord(self.exit_keybind) :
                break


        self.video_capture.release()
        cv2.destroyAllWindows()
        

    def morphing_realtime(self, img_model : Image) :
        
        if not(self.video_capture.isOpened()) :
            raise Exception("An error occured trying to access the WebCam")

        while(self.video_capture.isOpened()):

            retrieve, frame = self.video_capture.read()

            if not(retrieve) :
                raise Exception("An error occured during the recording")

            img_applied = Image(frame, mtcnn = self.mtcnn, dlib = self.dlib, model = False, title="Face_morphing")
           

            Image.morphing(img_model,img_applied)

            key = cv2.waitKey(self.delay)
     
            if key == ord(self.exit_keybind) :
                break


if __name__ == "__main__" :

    video = WebCamVideo()
    img_model = Image.get_test()
    #video.detect_face_live()
    video.morphing_realtime(img_model)





