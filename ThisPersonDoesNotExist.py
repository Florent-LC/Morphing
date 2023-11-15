import requests
import cv2
import os



class ThisPersonDoesNotExist :

    def __init__(self, save_file_name : str, color = True, title : str = "") :

        self.url = "https://thispersondoesnotexist.com/"
        self.save_file_name = save_file_name
        self.color = int(color)
        self.title = title

        try:
            response = requests.get(self.url)

            # Vérifie si la requête a réussi (code d'état 200)
            if response.status_code == 200:
                
                self.content = response.content
            
            else:
                raise Exception(f"Request failure with code {response.status_code}")
                
    
        except Exception as e:

            print (f"Failed to send the request : {e}")
            raise



    def save(self) :

        with open(self.save_file_name, 'wb') as f :
            f.write(self.content)



    def save_test() :

        person = ThisPersonDoesNotExist("test.jpg")
        person.save()



    def show(self) :

        img = cv2.imread(self.save_file_name, self.color)

        cv2.imshow(self.title, img)
 
        cv2.waitKey(0)
 
        cv2.destroyAllWindows()



    def show_test() :

        person = ThisPersonDoesNotExist("test2.jpg")
        person.save()
        person.show()

        # delete the file after showing it
        os.remove("test2.jpg")
    



if __name__ == "__main__" :
   
    ThisPersonDoesNotExist.save_test()
    ThisPersonDoesNotExist.show_test()
