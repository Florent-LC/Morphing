r"""
In order to run the test, use this command line, and make sure to be on the parent directory (.\Morphing):

python -m visual_test.Image_manipulation_test

"""

from facenet_pytorch import MTCNN
import dlib
import cv2
from code.Image_manipulation import Image


mtcnn = MTCNN(keep_all=True)
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_test_no_face(title: str):
    img = cv2.imread("no_face.png", 1)

    return Image(img, mtcnn, dlib_predictor, False, title=title)


def show_test():
    img_color = Image.get_face(
        "Test to show an image in color and in black and white"
    )

    img_color.show(False)

    # plot converted in grayscale
    img_color.show(False, False)


def rotation_test():
    img = Image.get_face("Test of a rotation")

    img.rotation(45.0)
    img.show(False)


def draw_circle_test():
    img = Image.get_face(
        "Test of the draw of two circles (one is filled and not the other one)"
    )

    img.draw_circle(img.center, 100, True)
    img.draw_circle(img.center, 200, False)

    img.show(False)


def draw_point_test():
    img = Image.get_face("Test to draw a point (filled)")

    img.draw_point(img.center)

    img.show(False)


def draw_rectangle_test():
    img = Image.get_face("Test to draw a rectangle")

    a, b = img.center
    img.draw_rectangle(img.center, (a + 300, b + 300))

    img.show(False)


def write_text_test():
    img = Image.get_face("Test to write some text on an image")

    img.write_text("Hello World ! / 42.22293556", img.center)

    img.show(False)


def detect_faces_test():
    img = Image.get_face("Test to detect face and draw a rectangle around it")

    img.detect_faces()
    img.show(False)


def detect_faces_test_no_face():
    img = get_test_no_face(
        "Test to show that there is no glitch when trying to detect a face when there is not"
    )

    img.detect_faces()

    img.show(False)


def set_landmarks_test():
    img = Image.get_face("Test to draw the landmark points")

    img.set_landmarks()
    for i, (x, y) in enumerate(img.landmarks_list):
        img.draw_point((x, y))
        img.write_text(str(i), (x, y))
    img.show(False)


def extract_face_test():
    img = Image.get_face("Test to extract the face")

    img.extract_face()
    cv2.imshow(img.title, img.face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def set_Delaunay_Triangulation_test():
    img = Image.get_face("Test to show the Delaunay triangulation")

    img.set_Delaunay_Triangulation()

    for i, t in enumerate(img.triangles):
        x1, y1 = t[0]
        x2, y2 = t[1]
        x3, y3 = t[2]

        img.write_text(
            f"{i}", (int((x1 + x2 + x3) / 3), int((y1 + y2 + y3) / 3)), 0.4
        )

        cv2.line(img.img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(img.img, (x2, y2), (x3, y3), (0, 0, 255), 2)
        cv2.line(img.img, (x1, y1), (x3, y3), (0, 0, 255), 2)

    img.show(False)


def morphing_test(debug: bool = False):
    img1 = Image.get_face("Test to show the morphing: first face")
    img2 = Image.get_face(
        "Test to show the morphing: second face", model=False
    )

    img1.show(False)
    img2.show(False)
    img_morphing = Image.morphing(img1, img2, debug)

    cv2.imshow("Morphing Image", img_morphing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


rotation_test()
draw_rectangle_test()
draw_circle_test()
draw_point_test()
write_text_test()
detect_faces_test()
detect_faces_test_no_face()
show_test()
for _ in range(2):  # verifying that the landmarks are the same
    set_landmarks_test()
extract_face_test()
for _ in range(2):  # verifying that the triangles are different
    set_Delaunay_Triangulation_test()
morphing_test()
