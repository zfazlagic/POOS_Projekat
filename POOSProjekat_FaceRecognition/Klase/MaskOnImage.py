import glob

import cv2
from PIL import Image
from PIL import ImageChops
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from skimage import io


def maskOnImage(path_to_masks, path_to_images, path_to_results):
    c = 0


    folderi = ['Robert Downey Jr', 'Other', 'Angelina Jolie']
    for i in folderi:
        vel = 0
        for filename in (path_to_images+ '/'+ str(i) +'/'+str(c)+ '/.JPG'):
            vel+= 1
        print(vel)
        j=0
        while(j!=vel):
            sl = cv2.imread(path_to_images + '/' + str(i) + '/' + str(c)+ '/.JPG')
            mask = cv2.imread(path_to_masks + '/' + str(i) + '/' + str(c) + '/.JPG' , 0)
            res = cv2.bitwise_and(sl, sl, mask=mask)
            cv2.imwrite(path_to_results + "/" + str(i) + "/" + str(c) + '/.JPG' ,res)
            c += 1
            j+=1


path_to_mask_folder = '../DataSetPOOS/Maske'
path_to_result = r'../DataSetPOOS/PrimjenaMaske'
path_to_original_images_folder = r'../DataSetPOOS/Images'

maskOnImage(path_to_mask_folder, path_to_original_images_folder, path_to_result)
