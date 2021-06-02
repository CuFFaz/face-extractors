#Importing Required Libraries
import os
import cv2 
import time

#Starting time counter
start = time.time()

#Specifying path of training data
root_dir = 'trainset'

#Empty List for storing our Extracted Faces
IMAGES = []

#Implementing our trained HaarClassifier for detecting frontal faces from our training data. 
kascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Iterating through directories & subdirectories and extracting the path of every jpg file present in "trainset" folder
#Outcome would be that the Extracted Face inside a particular image will be written in the same destination with the same name to that of the original file
for subdir, dirs,files in os.walk(root_dir):
    for file in files:
        #Reading/Loading the image
        img = cv2.imread(os.path.join(subdir, file))
        #resizing the image but keeping aspect ratio of the image unchanged (this is with.respect.to HEIGHT)
        #this is achieved by the formula (original width x new height)/ original height = new width
        height = 256
        width = img.shape[1]*height/img.shape[0]
        img = cv2.resize(img, (int(width), height), None, 0.5, 0.5, interpolation=cv2.INTER_AREA)
        #Conversion to a grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Feeding the grayscale image to our Classifier
        face = kascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
                )
        #HaarClassifier returned a 2D array, thus I'll extract these dimensions
        #and use the same dims to crop out the portion from our original image. Hence preserving our color channels.
        for (x,y,w,h) in face:
            #Slicing out the face as mentioned above
            crop = img[y:y+h,x:x+w]
            #Saving the image in the same folder with same names
            #thus replacing our previous image and keeping the Extracted Face image only            
            cv2.imwrite(os.path.join(subdir, file), crop)
            #Adding every extracted face to our empty list
            IMAGES.append(crop)


#Seizing Time Counter
end = time.time()

#Just some details for self understanding (Total Number of Faces extracted from the dataset, type of the variable 
#and total time taken to process)
print(len(IMAGES), type(IMAGES), ((start-end)/60))