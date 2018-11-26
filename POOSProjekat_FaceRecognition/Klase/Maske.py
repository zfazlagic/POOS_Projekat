from skimage import draw
from skimage import io
import numpy as np
import urllib.request
import json
import logging
import os
import sys

def convert_dataturks_to_masks(path_to_dataturks_annotation_json, path_to_original_images_folder, path_to_masks_folder):
    # make sure everything is setup.
    if (not os.path.isdir(path_to_original_images_folder)):
        logging.exception(
            "Please specify a valid directory path to download images, " + path_to_original_images_folder + " doesn't exist")
        return
    if (not os.path.isdir(path_to_masks_folder)):
        logging.exception(
            "Please specify a valid directory path to write mask files, " + path_to_masks_folder + " doesn't exist")
        return
    if (not os.path.exists(path_to_dataturks_annotation_json)):
        logging.exception(
            "Please specify a valid path to dataturks JSON output file, " + path_to_dataturks_annotation_json + " doesn't exist")
        return

    f = open(path_to_dataturks_annotation_json)
    train_data = f.readlines()
    train = []
    for line in train_data:
        data = json.loads(line)
        train.append(data)
    c = 0
    for objects in train:
        annotations = objects['annotation']

        for annot in annotations:
            label = str(annot['label'])
            
            urllib.request.urlretrieve(objects['content'], path_to_original_images_folder + "/" + str(c) + "/" + str(label) + ".jpg")
            if (label != ''):

                points = annot['points']
                h = annot['imageHeight']
                w = annot['imageWidth']
                x_coord = []
                y_coord = []
                l = []
                for p in points:
                    x_coord.append(p[0] * w)
                    y_coord.append(p[1] * h)
                shape = (h, w)
                mask = np.zeros((h, w, 3))
                row_coordinates, col_coordinates = draw.polygon(y_coord, x_coord, shape)
                mask[row_coordinates, col_coordinates] = 1

                io.imsave(path_to_masks_folder + "/" + str(c) + "/" + str(label) + ".jpg", mask)
        c += 1



path_to_mask_folder='..\DataSetPOOS\Maske'
path_to_dataturks_annotation_json=r'..\DataSetPOOS\Anotacije\DataSetPOOSFR.json'
path_to_original_images_folder=r'..\DataSetPOOS\Images'


convert_dataturks_to_masks(path_to_dataturks_annotation_json,path_to_original_images_folder,path_to_mask_folder)