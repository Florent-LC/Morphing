from random import triangular
from facenet_pytorch import MTCNN
import dlib
import cv2
import os
import numpy as np
from typing import Tuple

from ThisPersonDoesNotExist import *



mtcnn = MTCNN(keep_all=True)
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



class Image : 


    def __init__(self, img : np.ndarray, mtcnn : MTCNN, dlib : type[dlib_predictor], model : bool, title : str = "") :

        self.img = img
        # for some operations, the input has to be a black and white picture
        self.img_bw = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        # which image we consider : the model from which we apply the morphing (model = True)
        # or the destination (model = False)
        self.model = model
        
        self.title = title
        
        self.height,self.width = self.img.shape[:2]
        self.center = (int(self.width/2), int(self.height/2))
        
        # we use a MTCNN network to detect faces fast (faster than the detector provided
        # by dlib)
        self.mtcnn = mtcnn
        # we use a dlib detector to detect landmarks once the face is detected
        self.dlib = dlib
        
        # a parameter used when a face is detected
        # if no face is detected, the default value of this parameter is None
        self.box = None



    def get_test(model : bool = True) :

        person = ThisPersonDoesNotExist("test.jpg")
        person.save()
        img = cv2.imread(person.save_file_name,person.color)

        return Image(img, mtcnn, dlib_predictor, model, title = "Test")


    def get_test_no_face() :

        img = cv2.imread("no_face.png",1)

        return Image(img, mtcnn, dlib_predictor, False, title = "Test with no face")


    def show(self, continuous = True, color = True) :
        
        if color : 
            cv2.imshow(self.title, self.img)
        else :
            cv2.imshow(self.title, self.img_bw)
        if not(continuous) :
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def show_test() :

        img = Image.get_test()

        img.show(False)
        img.show(False,False)

        os.remove("test.jpg")


    def rotation(self, angle : float) :
        ''' Rotate the image '''

        rotate_matrix = cv2.getRotationMatrix2D(center=self.center, angle=angle, scale=1)
        rotated_image = cv2.warpAffine(src=self.img, M=rotate_matrix, dsize=(self.width, self.height))

        self.img = rotated_image


    def rotation_test() :

        img = Image.get_test()

        img.rotation(45.)
        img.show(False)

        os.remove("test.jpg")


    def draw_circle(self, circle_center : Tuple[int, int], radius : int, filled : bool, thickness : int = 3) :
        ''' Draw a red circle '''
        

        # A thickness of -1 denotes a filled circle
        thickness2 = -1 if filled else thickness

        cv2.circle(self.img, circle_center, radius, (0, 0, 255), thickness=thickness2, lineType=cv2.LINE_AA)


    def draw_circle_test() :

        img = Image.get_test()

        img.draw_circle(img.center,100,True)
        img.draw_circle(img.center,200,False)

        img.show(False)

        os.remove("test.jpg")
        

    def draw_point(self, coordinate : Tuple[int, int]) :
        self.draw_circle(coordinate, radius = 1, filled = True)


    def draw_point_test() :

        img = Image.get_test()

        img.draw_point(img.center)

        img.show(False)

        os.remove("test.jpg")


    def draw_rectangle(self, start_point : Tuple[int,int], end_point : Tuple[int,int], thickness : int = 3) :
        ''' Draw a red rectangle '''

        cv2.rectangle(self.img, start_point, end_point, (0, 0, 255), thickness=thickness, lineType=cv2.LINE_8)


    def draw_rectangle_test() :

        img = Image.get_test()

        a,b = img.center
        img.draw_rectangle(img.center,(a+300,b+300))

        img.show(False)

        os.remove("test.jpg")


    def write_text(self, text : str, position : Tuple[int,int], fontscale : float = 0.5) :
        ''' Write a text on the image at a give coordinate '''
        
        cv2.putText(self.img, text, position, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = fontscale, color = (255,255,255))


    def write_text_test() :

        img = Image.get_test()

        img.write_text("Hello World ! / 42.22293556", img.center)

        img.show(False)

        os.remove("test.jpg")


    def truncate(x : float, p : int) :
        power = 10**p
        return round(x*power)/power


    def detect_faces(self, threshold : float = 0.9) :
        ''' Draw rectangles and face points if a face is detected
        
            threshold : probability threshold defining whether a box is drawn when a face is detected
        '''

        # boxes : float list list of dimension (n,4) where n is the number of faces, and the
        # four elements corresponds to the edges of the rectangle (coordinates of the top
        # right corner and of the bottom left corner)

        # probs : float list (n) : probability of detection


        boxes, probs = self.mtcnn.detect(self.img, landmarks=False)

        if not(boxes is None) :

            for (box, prob) in zip(boxes, probs) :

                if prob > threshold : 

                    a,b,c,d = map(int,box)
                    self.draw_rectangle((a,b), (c,d))

                    prob_truncate = Image.truncate(prob, 3)*100
                    self.write_text(f"{prob_truncate}%", (int((a+c)/2),d+10))


    def detect_faces_test() :

        img = Image.get_test()

        img.detect_faces()
        img.show(False)

        os.remove("test.jpg")


    def detect_faces_test_no_face() :

        img = Image.get_test_no_face()

        img.detect_faces()

        img.show(False)



    def set_box_face(self, threshold : float = 0.9) :
        ''' Return the box around the face if detected (above the threshold),
            Only one face is extracted (with the highest probability) '''

        boxes, probs = self.mtcnn.detect(self.img, landmarks=False)

        if not(boxes is None) :

            prob = probs[0]

            if prob > threshold :
                
                # in order to manipulate dlib object, the box has to be of dlib type
                a,b,c,d = boxes[0].astype(int)
    
                self.box = dlib.rectangle(a,b,c,d)
             
        

    
    def set_landmarks(self) :
        
        ''' model is just a boolean telling if the image considered is the model or
            the other face where we want to morph '''
        
        self.set_box_face()
        
        if not (self.box is None) :

            # a dlib object
            landmarks = self.dlib(self.img,self.box)
            
            # converted into a list and a numpy array 
            # For some functions (convexHull), the argument has to be a sequence (hence converting into a list) 
            # for others (insert), it has to be a Matlike one, so we convert into a numpy array
            # (by default, the points are integers)
            self.landmarks_list = [(point.x,point.y) for point in landmarks.parts()]
            self.landmarks_array = np.array(self.landmarks_list)
            
            if self.model : 
                
                # eventually, we create a dictionnary from the list, whose keys are the coordinate and values
                # are the indexes in the list, if the face considered is the model
                # This will be useful in the future since we want to have an universal way to know which triangle
                # we consider (independently on the image considered)
                self.landmarks_dictionnary = {coordinate: index for index, coordinate in enumerate(self.landmarks_list)}

        
    
    def set_landmarks_test() :
        
        img = Image.get_test()

        img.set_landmarks()
        for i,(x,y) in enumerate(img.landmarks_list) :
            img.draw_point((x,y))
            img.write_text(str(i),(x,y))
        img.show(False)

        os.remove("test.jpg")
        


    def extract_face(self) :
        
        self.set_landmarks()

        if not (self.box is None) :        

            # initializing the mask
            self.mask = np.zeros_like(self.img_bw)
            
            # the convex hull of the landmarks
            self.convexhull = cv2.convexHull(self.landmarks_array)

            # the mask, which was previously a white image,
            # becomes a matrix with black pixels inside the convex hull
            cv2.fillConvexPoly(self.mask, self.convexhull, 255)

            # with this formulation, we keep all the values of the image
            # ('and' operation between the two same images), but only
            # across the mask, meaning that we extract the face of the image

            # if it is a model face, we simply apply this operation
            if self.model :
                self.face = cv2.bitwise_and(self.img, self.img, mask=self.mask)
            # in the other case, we apply this operation for the complementary of the mask
            else :
                self.inverted_mask = cv2.bitwise_not(self.mask)
                self.face = cv2.bitwise_and(self.img, self.img, mask=self.inverted_mask)
            

    
    def extract_face_test() :

        img = Image.get_test()

        img.extract_face()
        cv2.imshow("Extracted Face", img.face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
        os.remove("test.jpg")
        


    def set_Delaunay_Triangulation(self) :

        self.extract_face()

        if not(self.box is None) :

            # the rectangle (cv2 object) bounding the convex hull
            self.rect = cv2.boundingRect(self.convexhull)
            
            # the part of the image that this rectangle bounds
            self.subdiv = cv2.Subdiv2D(self.rect)
            
            # inserting the landmarks on the subdiv
            self.subdiv.insert(self.landmarks_list)
            
            # computing the Delaunay triangulation by creating the triangles whose
            # vertexes are the landmarks
            # we need to have dtype=np.int32 since a function of opencv (boundingRect)
            # that needs this type
            self.triangles = np.array(self.subdiv.getTriangleList(), dtype=np.int32).reshape((-1,3,2))
            
            # now, we have the triangles whose values are the coordinate of each landmark
            # we will then transform these coordinates into their index, using landmarks_dictionnary
            # reminder : triangles is a n_triangles x 6 matrix, where the 6 values are x1,y1,x2,y2,x3,y3
            # the three coordinates of the vertexes of each triangle
            indexes_triangles = [ [self.landmarks_dictionnary[tuple(e[0])],
                                   self.landmarks_dictionnary[tuple(e[1])],
                                   self.landmarks_dictionnary[tuple(e[2])] ] for e in self.triangles ]
            self.indexes_triangles = np.array(indexes_triangles, dtype=np.uint8)
            
            


    def set_Delaunay_Triangulation_test() :
        
        img = Image.get_test()
        
        img.set_Delaunay_Triangulation()
        
        for i,t in enumerate(img.triangles):
            x1,y1 = (t[0], t[1])
            x2,y2 = (t[2], t[3])
            x3,y3 = (t[4], t[5])
            
            img.write_text(f"{i}",(int((x1+x2+x3)/3),int((y1+y2+y3)/3)),0.4)
            
            cv2.line(img.img, (x1,y1), (x2,y2), (0, 0, 255), 2)
            cv2.line(img.img, (x2,y2), (x3,y3), (0, 0, 255), 2)
            cv2.line(img.img, (x1,y1), (x3,y3), (0, 0, 255), 2)
            
        img.show(False)
        
        os.remove("test.jpg")
        


    def morphing(img_model : 'Image', img_applied : 'Image', debug : bool = False) :

        ''' Plot the resulting morphing (doesn't return anything) '''

        assert img_model.model, ("The first image should be the model")
        assert not(img_applied.model), ("The second image should be the destination")
        
        # creating the landmarks and the triangulation
        img_model.set_Delaunay_Triangulation()
        
        # creating the landmarks and extracting the face only (no need to compute the delaunay triangulation)
        img_applied.extract_face()
        
        # setting the triangles of the second face, by putting them in the same order as the triangles
        # of the first face
        img_applied.triangles = img_applied.landmarks_array[img_model.indexes_triangles]
        
        # the resulting morphing, of the shape of the second image
        img_morphing = np.zeros_like(img_applied.img)
        

        # now that the triangulation is done for both of the faces, we can perform the morphing,
        # by looping on all the triangles of both faces
        for i in range(len(img_model.triangles)) :

            # first face

            # the triangle of the first face
            triangle1 = img_model.triangles[i]

            # we restrict the study to the bounding rectangle
            (x, y, w, h) = cv2.boundingRect(triangle1)
            
            # the resulting cropped triangle, which is initialized with all the sub-window
            cropped_triangle1 = img_model.img[y: y + h, x: x + w]
            
            # the mask which will create the triangle from the window
            cropped_triangle_mask1 = np.zeros((h, w), np.uint8)

            # we transform the triangle so that it becomes an array
            # of the coordinate of the vertexes withing the bounding box
            # thanks to this, we can use the mask (which is just a zeros 
            # matrix of dimension hxw)
            points1 = triangle1 - np.array([x,y])
            
            # creating the mask by filling the triangle whose vertexes are the 'points'
            # so inside the triangle, the values are 255, and outside 0
            cv2.fillConvexPoly(cropped_triangle_mask1, points1, color=255)
            
            # extracting the face within the rectangle by applying the mask (black outside)
            cropped_triangle1 = cv2.bitwise_and(cropped_triangle1, 
                                                cropped_triangle1,
                                                mask=cropped_triangle_mask1)

          
            if debug : 
                cv2.imshow("Cropped triangle of the model", cropped_triangle1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


            # second face
            # for the second face, as we paste the first face on it, we don't need to 
            # consider the cropped triangle with the part of the face, but only its shape
            # so we are only interested in a mask
            # this mask will only be used to remove the edges of the resulting triangle

            triangle2 = img_applied.triangles[i]

            (x, y, w, h) = cv2.boundingRect(triangle2)
            cropped_triangle_mask2 = np.zeros((h, w), np.uint8)

            points2 = triangle2 - np.array([x,y])
            
            cv2.fillConvexPoly(cropped_triangle_mask2, points2, color = 255)


            # applying the morphing
            
            # we transform the type of the points to fit a specific function which requires float numbers
            # (getAffineTransform)
            points1 = np.float32(points1)
            points2 = np.float32(points2)
            

            # now, we transform the triangle of the first face, so that its shape is like the one of the second face
            # to do this, we use functions of opencv which tackles this interpolation problem

            # the matrix of the warping
            matrix_warping = cv2.getAffineTransform(points1, points2)
            # transforming the triangle using this matrix, filling with black the rest of the window
            warped_triangle = cv2.warpAffine(cropped_triangle1, matrix_warping, (w, h))
                    

            # we set the pixels on the border of the triangle to 0
            # so at this point, strictly inside the triangle, we have the part of the face considered,
            # and outside the triangle and on the border we have 0
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_triangle_mask2)

            
            if debug : 
                cv2.imshow("Warped triangle", warped_triangle)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


            # Reconstructing destination face

            # Selecting the box in which the triangle exists
            img_morphing_rect_area = img_morphing[y: y + h, x: x + w]
            

            # now we deal with the edges of the triangulation
            # if we didn't tackle this issue, we would have a morphing but the edges of the triangulation
            # would appear, making the morphing much less realistic
            
            # first, we get the part of the morphing we consider in black and white, in order to create a mask
            img_morphing_rect_area_gray = cv2.cvtColor(img_morphing_rect_area, cv2.COLOR_BGR2GRAY)
            
            # we create a binary mask that we will apply to the triangle : 0 for pixel values of the reconstructing
            # face that we have already explored (including some edges of the triangle), and 255 for non-explored pixels
            _,mask_triangles_designed = cv2.threshold(img_morphing_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)

            # applying the mask, we keep the triangle, but only for the pixels we haven't explored yet
            # (meaning we exclude the edges we have already filled for previous triangles)
            # if we didn't do that, the edges that we haven't explored yet would have never been filled
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
            
            # Adding the warped triangle to the result : outside the triangle, we have 0 for the triangle
            # and 0 or already filled pixels for img_morphing. Inside the triangle, we have face pixels
            # for the triangle, and 0 for img_morphing since we haven't filled this triangle yet
            img_morphing[y: y + h, x: x + w] = cv2.add(img_morphing_rect_area, warped_triangle)


            if debug :
                cv2.imshow("Morphing", img_morphing)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
           
    
        # now that we have the morphing, we paste it instead of the previous face
        img_morphing = cv2.add(img_applied.face,img_morphing)
        
        # in order to smooth the result of the morphing, we need to compute the center of the face
        (x, y, w, h) = cv2.boundingRect(img_applied.convexhull)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

        # an opencv function automatically smoothen the result
        res = cv2.seamlessClone(img_morphing, img_applied.img, img_applied.mask, center_face2, cv2.NORMAL_CLONE)
        
        cv2.imshow("Morphing Image", res)
        
            
            

    def morphing_test(debug : bool = False) :
        
        img1 = Image.get_test()
        img2 = Image.get_test(model=False)
                
        #img1.show(False)
        #img2.show(False)
        Image.morphing(img1,img2,debug)
        
        os.remove("test.jpg")





if __name__ == "__main__" :

    # Image.show_test()
    # Image.rotation_test()
    # Image.draw_rectangle_test()
    # Image.draw_circle_test()
    # Image.draw_point_test()
    # Image.write_text_test()
    # Image.detect_faces_test()
    # Image.detect_faces_test_no_face()
    # for _ in range(2) : # verifying that the landmarks are the same
    #     Image.set_landmarks_test()
    # Image.extract_face_test()
    # for _ in range(2) : # verifying that the triangles are different
    #     Image.set_Delaunay_Triangulation_test()
    Image.morphing_test()

           


