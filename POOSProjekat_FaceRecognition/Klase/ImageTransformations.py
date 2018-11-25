import cv2
from matplotlib import pyplot as plt
from PIL import Image


def convertToBlackAndWhite():
    image_file = Image.open("test.png") # open colour image
    image_file = image_file.convert('1') # convert image to black and white
    image_file.save('result.png')
    return image_file


# Metoda koja se koristi za maskiranje neostrina
def maskiranje_neostrina():
    image = cv2.imread(r"C:\Users\Lusi\Desktop\POOS_Projekat\POOSProjekat_FaceRecognition\DataSetPOOS\download.jpg")
    gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
    cv2.imwrite(r"C:\Users\lusi\Desktop\POOS_Projekat\POOSProjekat_FaceRecognition\DataSetPOOS\EditovaneSlike\unsharp_download.jpg", unsharp_image)
    return unsharp_image
# Metoda za poboljsavanje svjetlosti na slici

def increase_brightness(value=30):
    img = cv2.imread(r"C:\Users\lusi\Desktop\POOS_Projekat\POOSProjekat_FaceRecognition\DataSetPOOS\download.jpg");
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('C:/Users/lusi/Desktop/POOS_Projekat/POOSProjekat_FaceRecognition/DataSetPOOS/EditovaneSlike/editBrightness.jpg', img)
    return img

#Poboljsavanje kontrasta i ujednacaavnje histograma
def edit_contrast():
#-----Ucitavanje slike-----------------------------------------------------
    img = cv2.imread(r"C:\Users\lusi\Desktop\POOS_Projekat\POOSProjekat_FaceRecognition\DataSetPOOS\download.jpg", 1)
    cv2.imshow("img", img)

#-----Konverzija slike u LAB Color model-----------------------------------
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cv2.imshow("lab",lab)

#-----Rastavljanje LAB slike u razlicite kanale-------------------------
    l, a, b = cv2.split(lab)
    cv2.imshow('l_channel', l)
    cv2.imshow('a_channel', a)
    cv2.imshow('b_channel', b)

#-----Primjena CLAHE za L-kanal-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    cv2.imshow('CLAHE output', cl)

#-----Spajanje CLAHE poboljsanog L-channel sa a i b kanalima-----------
    limg = cv2.merge((cl,a,b))
    cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imshow('final', final)
    cv2.imwrite('C:/Users/lusi/Desktop/POOS_Projekat/POOSProjekat_FaceRecognition/DataSetPOOS/EditovaneSlike/editContrast.jpg', final)
    return final
#_____END_____#

#Uklanjanje šuma
def image_denoise():

    img = cv2.imread(r"C:\Users\lusi\Desktop\POOS_Projekat\POOSProjekat_FaceRecognition\DataSetPOOS\Downey.jpg")

    blur = cv2.bilateralFilter(img, 9, 75, 75)

    cv2.imwrite('C:/Users/lusi/Desktop/POOS_Projekat/POOSProjekat_FaceRecognition/DataSetPOOS/EditovaneSlike/blur.jpg', blur)
    return

# Pozivanje funkcija


increase_brightness()
maskiranje_neostrina()
edit_contrast()
image_denoise()