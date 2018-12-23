from skimage import io
import os

import cv2


def maskiranje_neostrina(path):
    image = cv2.imread(path)
    gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
    #cv2.imwrite(r"..\DataSetPOOS\EditovaneSlike\unsharp_download.jpg", unsharp_image)
    return unsharp_image

path = r"../DataSetPOOS/training-data"
dirs = os.listdir(path)
for dir in dirs:

    images_names = os.listdir(path+ "/" +dir)
    print(images_names)

    for image_name in images_names:
        image_path = path + "/" + dir+ "/"+image_name
        image = cv2.imread(image_path)
        poboljsanje = maskiranje_neostrina(image_path)
        path2 = r"../DataSetPOOS/PoboljsaneSlike"
        cv2.imwrite(path2 + "/" + dir + "/" + image_name, poboljsanje)


