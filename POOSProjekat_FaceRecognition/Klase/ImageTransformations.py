import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

#Ukoliko bude potrebno funkcija za konvertovanje slike u boji u B&W

def convertToBlackAndWhite():
    image_file = Image.open("test.png") # open colour image
    image_file = image_file.convert('1') # convert image to black and white
    image_file.save('result.png')
    return image_file


# Metoda koja se koristi za maskiranje neostrina

def maskiranje_neostrina():
    image = cv2.imread(r"..\DataSetPOOS\Images\download.jpg")
    gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
    cv2.imwrite(r"..\DataSetPOOS\EditovaneSlike\Images\unsharp_download.jpg", unsharp_image)
    return unsharp_image

# Metoda za poboljsavanje svjetlosti na slici

def increase_brightness(value=30):
    img = cv2.imread(r"..\DataSetPOOS\Images\download.jpg");
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('../DataSetPOOS/EditovaneSlike/Images/editBrightness.jpg', img)
    return img

#Poboljsavanje kontrasta i ujednacaavnje histograma

def clahe():
#-----Ucitavanje slike-----------------------------------------------------
    img = cv2.imread(r"..\DataSetPOOS\Images\download.jpg", 1)
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

#-----Konvertovanje slike iz LAB Color modela u RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imshow('final', final)
    cv2.imwrite('../DataSetPOOS/EditovaneSlike/Images/editHistogram.jpg', final)
    return final


#Uklanjanje šuma

def image_denoise():

    img = cv2.imread(r"..\DataSetPOOS\Images\rdj.jpeg")

    blur = cv2.bilateralFilter(img, 10, 75, 75)

    cv2.imwrite('../DataSetPOOS/EditovaneSlike/Images/blur.jpg', blur)
    return

# Poboljšavanje kontrasta

def edit_contrast():
    image = Image.open(r"..\DataSetPOOS\Images\download.jpg")
    scale_value = 2.5

    image = ImageEnhance.Contrast(image).enhance(scale_value)
    image.save('../DataSetPOOS/EditovaneSlike/Images/editContrast.jpg')
    return


# Pozivanje funkcija

increase_brightness()
maskiranje_neostrina()
edit_contrast()
image_denoise()
clahe()