from skimage import io
import os

import cv2


def maskiranje_neostrina(path):
    image = cv2.imread(path)
    gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.4, 0, image)
    #cv2.imwrite(r"..\DataSetPOOS\EditovaneSlike\unsharp_download.jpg", unsharp_image)
    return unsharp_image

path = r"../test/"
dirs = os.listdir(path)

images_names = os.listdir(path)
print(images_names)

for image_name in images_names:
    image_path = path + "/"+image_name
    image = cv2.imread(image_path)
    poboljsanje = maskiranje_neostrina(image_path)
    path2 = r"../TestPoboljsanje"
    cv2.imwrite(path2 + "/" + image_name, poboljsanje)


