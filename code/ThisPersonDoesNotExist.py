# -*- coding: utf-8 -*-

"""

The file used to get a randomly generated face


"""


import requests
import cv2
import numpy as np


class ThisPersonDoesNotExist:
    def __init__(self, save_file_name: str = "", color=True, title: str = ""):
        self.url = "https://thispersondoesnotexist.com/"
        self.save_file_name = save_file_name
        self.color = int(color)
        self.title = title

        try:
            response = requests.get(self.url)

            # Checks if the request was successful (status code 200)
            if response.status_code == 200:
                self.content = response.content

            else:
                raise Exception(
                    f"Request failure with code {response.status_code}"
                )

        except Exception as e:
            print(f"Failed to send the request : {e}")
            raise

    def decode_image(self):
        """Transform the content of self (of type 'bytes') into a numpy array of pixels"""

        img_np = np.frombuffer(self.content, np.uint8)

        self.img = cv2.imdecode(img_np, self.color)

    def save(self):
        """The function 'write' doesn't need to save a numpy array, it can translate a byte and
        automatically detect that it should be save as a jpg file"""

        with open(self.save_file_name, "wb") as f:
            f.write(self.content)

    def show(self):
        img = cv2.imread(self.save_file_name, self.color)

        cv2.imshow(self.title, img)

        cv2.waitKey(0)

        cv2.destroyAllWindows()
