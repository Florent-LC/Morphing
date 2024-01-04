r"""
In order to run the test, use this command line, and make sure to be on the parent directory (.\Morphing):

python -m visual_test.ThisPersonDoesNotExist_test

"""
import os
from code.ThisPersonDoesNotExist import ThisPersonDoesNotExist


def show_test():
    person = ThisPersonDoesNotExist("test2.jpg")
    person.save()
    person.show()

    # delete the file after showing it
    os.remove("test2.jpg")


# it should plot a randomly generated face
show_test()
