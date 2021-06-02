#Importing Required Libraries
import os
import time
import face_recognition
from PIL import Image

#Starting time counter
start = time.time()

#Specifying path of training data
root_dir = 'trainset'

#Empty List for storing our Extracted Faces just for fun and info
IMAGES = []

#Iterating through directories & subdirectories and extracting the path of every jpg file present in "trainset" folder
#Outcome would be that the Extracted Face inside a particular image will be written in the same destination with the same name to that of the original file
for subdir, dirs,files in os.walk(root_dir):
    for file in files:
    #Loading the Image file, it is stored in the form of a numpy array.
        img = face_recognition.load_image_file(os.path.join(subdir, file))
        #Extracting the face locations from the array based image
        face_locs = face_recognition.face_locations(img)
        #face_recognition returns an array of the face locations, 
        #hence I'll use the same dims to crop out the portion from our original image.
        for (top, right, bottom, left) in face_locs:
            crop = img[top:bottom, left:right]
            #Convert the face image to a PIL-format image so that we can save it in the same folder with same names
            #thus replacing our previous image and keeping the Extracted Face image only
            pil_image = Image.fromarray(crop)
            #saving the resulting face image
            pil_image.save(os.path.join(subdir, file))
            #Adding every resulting extracted face to our empty list
            IMAGES.append(pil_image)

#Seizing the counter
end = time.time()
#Just some details for self understanding (Total Number of Faces extracted from the dataset, type of the variable 
#and total time taken to process)
print(len(IMAGES), type(IMAGES), ((end-start)/60))