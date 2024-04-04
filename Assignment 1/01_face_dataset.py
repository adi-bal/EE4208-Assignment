import cv2
import os

dataset_path = r"C:\Users\Bryan\Desktop\EE4208_DATASET"
cascade_classifier_path = r"C:\Users\Bryan\Desktop\EE4208 Assignment 1\EE4208-Assignment\Assignment 1\haarcascade_frontalface_default.xml"
count = 0

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier(cascade_classifier_path)

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count += 1
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     

        # Generating filepath to save the captured images into
        savefilepath = os.path.join(dataset_path, str(face_id) + "." + "1" + "." + str(count) + ".jpg")
        print(savefilepath)

        # Save the captured image into the datasets folder
        cv2.imwrite(savefilepath, gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


