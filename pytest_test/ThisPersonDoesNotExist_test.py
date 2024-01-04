r"""
In order to run the test, use this command line, and make sure to be on the parent directory (.\Morphing):

"python -m pytest .\pytest_test\ "

There is a bug with Pytest: it indicates "no tests ran", even if the tests are actually done
(It gives an error when the assertion is reversed (not(...)))

"""

from code.ThisPersonDoesNotExist import ThisPersonDoesNotExist
import os
import numpy as np


def decode_image_test():
    person = ThisPersonDoesNotExist("test.jpg", color=True)
    person.decode_image()

    assert isinstance(
        person.img, np.ndarray
    ), f"The image of the person is not a numpy array, but of type {type(person.img)}"
    assert (
        person.img.ndim == 3
    ), f"The number of dimension {person.img.ndim} should be 3 since the image is in color"

    person = ThisPersonDoesNotExist("test.jpg", color=False)
    person.decode_image()

    assert isinstance(
        person.img, np.ndarray
    ), f"The image of the person is not a numpy array, but of type {type(person.img)}"
    assert (
        person.img.ndim == 2
    ), f"The number of dimension {person.img.ndim} should be 2 since the image is in black and white"


def save_test():
    # deleting the potentially already existing file "test.jpg"
    if os.path.exists("test.jpg"):
        os.remove("test.jpg")

    person = ThisPersonDoesNotExist("test.jpg")
    person.save()

    assert os.path.exists(
        "test.jpg"
    ), "The file 'test.jpg' should exist after the saving"

    os.remove("test.jpg")


decode_image_test()
save_test()
