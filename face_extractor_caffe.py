#Importing Required Libraries
import os
import cv2 
import numpy as np
import time

#Starting time counter
start = time.time()

#Specifying paths of training data
root_dir = 'trainset'

#Specifying paths to Model architecture and Pretrained Weights
prototxt_path = 'model_data/deploy.prototxt'
caffemodel_path = 'model_data/weights.caffemodel'	

#Empty Var for storing our Extracted Faces just for fun and info
IMAGES = []

#Implementing our pretrained DNN with Caffe Model for detecting frontal faces from our training data. 
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

i = 1

#Iterating through directories & subdirectories and extracting the path of every jpg file present in "trainset" folder
#Outcome would be that the Extracted Face inside a particular image will be written in the same destination with the same name to that of the original file
for subdir, dirs,files in os.walk(root_dir):
    for file in files:
        #Reading/Loading the image
        image = cv2.imread(os.path.join(subdir, file))
        #Grabbing image dimensions
        (h, w) = image.shape[:2]
        #Resizing image to a specific dimension and creating a blob
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (256, 256)), 1.0, (256, 256), (104.0, 177.0, 123.0))
        #Feeding the blob to our pretrained DNN
        model.setInput(blob)
        detections = model.forward()
        #Looping through every single face detection in our image
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            #getting the dims of the extracted box containing our face
            (startX, startY, endX, endY) = box.astype("int")
            #grabbing the confidence value of the predicted box that its a face
            confidence = detections[0, 0, i, 2]
            #Only those Detections with more than 50% confidence will be extracted
            if (confidence > 0.5):
                #Slicing out the face
                crop = image[startY:endY, startX:endX]
                #Exception created for empty images. 
                #7-10 recurrent empty images were found after image processing done above thus had to exclude them
                try:
                    #Saving the image in the same folder with same names
                    #thus replacing our previous image and keeping the Extracted Face image only                    
                    cv2.imwrite(os.path.join(subdir, file), crop)
                    IMAGES.append(crop)
                except:
                    #exception of empty images that couldnt be written/saved. (Got 7-10 such images, couldnt understand why facing issue with this method)
                    print(f"Number of Empty Ones found:-{i}")
                    i+=1

#Seizing Time Counter
end = time.time()

#Just some details for self understanding (Total Number of Faces extracted from the dataset, type of the variable 
#and total time taken to process)
print(len(IMAGES), type(IMAGES), ((start-end)/60))