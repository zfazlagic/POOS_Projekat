import cv2

# Metoda koja se koristi za maskiranje neostrina

image = cv2.imread(r"C:\Users\mali_cox\Desktop\POOS_Projekat\POOSProjekat_FaceRecognition\DataSetPOOS\download.jpg")
gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0)
unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
cv2.imwrite(r"C:\Users\mali_cox\Desktop\POOS_Projekat\POOSProjekat_FaceRecognition\DataSetPOOS\EditovaneSlike\unsharp_download.jpg", unsharp_image)

# Metoda za poboljsavanje svjetlosti na slici

def increase_brightness(value=30):
    img = cv2.imread(r"C:\Users\mali_cox\Desktop\POOS_Projekat\POOSProjekat_FaceRecognition\DataSetPOOS\download.jpg");
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('C:/Users/mali_cox/Desktop/POOS_Projekat/POOSProjekat_FaceRecognition/DataSetPOOS/EditovaneSlike/editBrightness.jpg', img)
    return img

increase_brightness()