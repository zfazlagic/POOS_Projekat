from skimage import draw
from skimage import io
import numpy as np
import urllib.request
import json
import os

def poly2mask(blobs, c, path_to_masks_folder, h, w, label, idx):
    mask = np.zeros((h, w))
    for l in blobs:
        fill_row_coords, fill_col_coords = draw.polygon(l[1], l[0], l[2])
        mask[fill_row_coords, fill_col_coords] = 1
    io.imsave(path_to_masks_folder + "/" + str(label) + "/" + str(c) + ".jpg", mask, mask)

path_to_mask_folder='..\DataSetPOOS\Maske'

def kreirajMasku(path_to_annotation, path_to_images, path_to_masks):
    if (not os.path.isdir(path_to_images)):
        print("Folder {} ne postoji.".format(path_to_images))
        return
    if (not os.path.isdir(path_to_masks)):
        print("Folder {} ne postoji.".format(path_to_masks))
        return
    if (not os.path.exists(path_to_annotation)):
        print("Folder {} ne postoji.".format(path_to_annotation))
        return

    f = open(path_to_annotation)
    lines = f.readlines()
    data = []
    for line in lines:
        d = json.loads(line)
        data.append(d)
    c = 0
    for d in data:
        classes={}
        annotations = d['annotation']

        for annotation in annotations:
            label = str(annotation['label'])
            label = label[2:-2]
            urllib.request.urlretrieve(d['content'],
                                           path_to_images + "/" + str(label) + "/" + str(
                                               c) + ".jpg")





            if label != '':
                if label not in classes:
                    classes[label] = 0
                blobs = []
                points = annotation['points']
                h = annotation['imageHeight']
                w = annotation['imageWidth']
                x_coord = []
                y_coord = []
                l = []
                for point in points:
                    x_coord.append(point[0] * h)
                    y_coord.append(point[1] * w)
                shape = (h, w)
                l.append(y_coord)
                l.append(x_coord)
                l.append(shape)
                blobs.append(l)
                mask = np.zeros((h, w))
                for l in blobs:
                    fill_row_coords, fill_col_coords = draw.polygon(l[0], l[1],l[2])
                    mask[fill_row_coords, fill_col_coords] = 1
                io.imsave(path_to_masks + "/" + str(label) + "/" + str(c) + ".jpg", mask)
                classes[label] += 1
                #mask = np.zeros((h, w, 3))
                #row_coordinates, col_coordinates = draw.polygon(x_coord, y_coord, shape)
                #mask[row_coordinates, col_coordinates] = 1

                #io.imsave(path_to_masks + "/" + str(label) + "/" + str(c) + ".jpg", mask)
        c += 1



path_to_mask_folder=r'..\DataSetPOOS\Maske'
path_to_dataturks_annotation_json=r'..\DataSetPOOS\Anotacije\DataSetPOOSFR.json'
path_to_original_images_folder=r'..\DataSetPOOS\Images'


kreirajMasku(path_to_dataturks_annotation_json,path_to_original_images_folder,path_to_mask_folder)