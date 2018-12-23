import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

#Ukoliko bude potrebno funkcija za konvertovanje slike u boji u B&W

def convertToBlackAndWhite(path):
    image_file = Image.open(path) # open colour image
    image_file = image_file.convert('1') # convert image to black and white
    #image_file.save('result.png')
    return image_file


# Metoda koja se koristi za maskiranje neostrina

def maskiranje_neostrina(path):
    image = cv2.imread(path)
    gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
    #cv2.imwrite(r"..\DataSetPOOS\EditovaneSlike\unsharp_download.jpg", unsharp_image)
    return unsharp_image

# Metoda za poboljsavanje svjetlosti na slici

def increase_brightness(path):
    value=30
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    #cv2.imwrite('../DataSetPOOS/EditovaneSlike/editBrightness.jpg', img)
    return img

#Poboljsavanje kontrasta i ujednacaavnje histograma

def clahe(path):
#-----Ucitavanje slike-----------------------------------------------------
    img = cv2.imread(path, 1)
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
    #cv2.imwrite('../DataSetPOOS/EditovaneSlike/editHistogram.jpg', final)
    return final


#Uklanjanje šuma

def image_denoise(path):

    img = cv2.imread(path)

    blur = cv2.bilateralFilter(img, 5, 15, 15)

    #cv2.imwrite('../DataSetPOOS/EditovaneSlike/blur.jpg', blur)
    return blur

# Poboljšavanje kontrasta

def edit_contrast(path):
    image = Image.open(path)
    scale_value = 1

    image = ImageEnhance.Contrast(image).enhance(scale_value)
    #image.save('../DataSetPOOS/EditovaneSlike/editContrast.jpg')
    return image


# Pozivanje funkcija
############################################
#path=r"..\DataSetPOOS\training-data\s0\21.jpg"
#increase_brightness()
#image=maskiranje_neostrina(path)
#cv2.imwrite(r"..\DataSetPOOS\EditovaneSlike\unsharp_download.jpg",image)
#edit_contrast()
#image_denoise()
#clahe()
#convertToBlackAndWhite(path)
#########################################