import cv2
import os
import numpy as np

subjects = ["Other", "Robert Downey Jr.", "Angelina Jolie"]

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def detect_face(img):
    # Slike se pretvaraju u gray-scale .OpenCV detektor radi sa gray-scale slikama
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Koristenje OpenCV detektora, ovdje se koristi LBP jer je brz
    # Moze se koristiti Haar koji je sporiji
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    # let's detect multiscale (some images may be closer to camera than others) images
    # Rezultat je lista lica
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=5)

    # Slucaj da nije otkriveno lice
    if (len(faces) == 0):
        return None, None

    # Ukoliko postoji samo jedno lice ekstraktovati podrucje lica
    (x, y, w, h) = faces[0]

    # vratiti samo dio slike oko lica
    return gray[y:y + w, x:x + h], faces[0]


def importYML(test_img):
    faceDetect=cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    cam=cv2.VideoCapture(0)
    rec=cv2.face.LBPHFaceRecognizer_create()
    rec.read("../trainner/trainner.yml")

    img = test_img.copy()
    face, rect = detect_face(img)

    label, confidence = rec.predict(face)
    # get name of respective label returned by face recognizer
    label_text = subjects[label]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.03,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        draw_text(img, label_text, rect[0], rect[1] - 5)
        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255);


    cam.release()
    cv2.destroyAllWindows()
    return img

path = "../test/"
dirs = os.listdir(path)
images_names = os.listdir(path)
print(images_names)
slike = []
for image_name in images_names:
    image_path = path + "/" + image_name
    image = cv2.imread(image_path)
    predicted_img = importYML(image)
    slike.append(predicted_img)

for slika in slike:
    cv2.imshow("Face classified", cv2.resize(slika, (400, 500)))
    cv2.waitKey(0)