r"""
In order to run the this file, use this command line, and make sure to be on the parent directory (.\Morphing):

python -m run.Run_webcam

"""

from code.Webcam import WebCamVideo
from code.Image_manipulation import Image


# whether to simply detect the face in realtime for the run
detect_face = True

# wether to morph in realtime the face with a randomly generated face
morphing = True

fps = 30.0
# echap keybind
exit_keybind = "\x1b"


def main():
    if detect_face:
        video = WebCamVideo(fps=fps, exit_keybind=exit_keybind)
        video.detect_face_live()

    if morphing:
        video = WebCamVideo(fps=fps, exit_keybind=exit_keybind)
        img_model = Image.get_face(title="Generated Face", model=True)
        video.morphing_realtime(img_model)


if __name__ == "__main__":
    main()
