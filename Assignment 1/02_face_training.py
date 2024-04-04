
import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
dataset_path = r"C:\Users\Bryan\Desktop\EE4208_DATASET"

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(r"C:\Users\Bryan\Desktop\EE4208 Assignment 1\EE4208-Assignment\Assignment 1\haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(dataset_path):

    imagePaths = [os.path.join(dataset_path,f) for f in os.listdir(dataset_path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(dataset_path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.save(r"C:\Users\Bryan\Desktop\trainer.yml")

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
