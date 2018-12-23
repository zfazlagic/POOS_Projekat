import cv2
import os

# Get user supplied values
folderPath = "../test"
cascPath = "opencv-files/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

subject_images_names = os.listdir(folderPath)

for image_name in subject_images_names:
    image = cv2.imread(folderPath+"/"+image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5,)

    print("Found {0} face(s)!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)



