from facenet_pytorch import MTCNN
import dlib
import cv2
from code.Image_manipulation import Image


mtcnn = MTCNN(keep_all=True)
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


class WebCamVideo:
    def __init__(self, fps: float = 30.0, exit_keybind: str = "\x1b"):
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.fps = fps

        # the delay between two frames in ms
        self.delay = int(1000 / self.fps)
        self.exit_keybind = exit_keybind  # "Esc" keybind

        self.mtcnn = MTCNN(keep_all=False)
        self.dlib = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat"
        )

    def detect_face_live(self):
        if not (self.video_capture.isOpened()):
            raise Exception("An error occured trying to access the WebCam")

        while self.video_capture.isOpened():
            retrieve, frame = self.video_capture.read()

            if not (retrieve):
                raise Exception("An error occured during the recording")

            img = Image(
                frame,
                mtcnn=self.mtcnn,
                dlib=self.dlib,
                model=False,
                title="Face_detection",
            )
            img.detect_faces()
            img.show()

            key = cv2.waitKey(self.delay)

            if key == ord(self.exit_keybind):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()

    def morphing_realtime(self, img_model: Image):
        if not (self.video_capture.isOpened()):
            raise Exception("An error occured trying to access the WebCam")

        while self.video_capture.isOpened():
            retrieve, frame = self.video_capture.read()

            if not (retrieve):
                raise Exception("An error occured during the recording")

            img_applied = Image(
                frame, mtcnn=self.mtcnn, dlib=self.dlib, model=False
            )
            img_morphing = Image.morphing(img_model, img_applied)

            if not (img_morphing is None):
                cv2.imshow("Morphing Image", img_morphing)

            key = cv2.waitKey(self.delay)

            if key == ord(self.exit_keybind):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()
