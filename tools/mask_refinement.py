import os
import glob

import cv2
import numpy as np
import segmentation_refinement as refine


image_paths_tif='/path/to/queryset/images'
label_pths_png = glob.glob('results/preds' + '/*.png')
OUT_DIR_b1 = 'post-process/cascadepsp/building_type_1'
OUT_DIR_b2 = 'post-process/cascadepsp/building_type_2'

refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

for fn_img in label_pths_png:
    mask = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)  
    filename = os.path.splitext(os.path.basename(fn_img))[0]
    image = cv2.imread(os.path.join(image_paths_tif, filename + '.tif'))
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    mash_num = 7
    mask[mask != mash_num] = 0
    mask[mask == mash_num] = 255
   
    output = refiner.refine(image, mask, fast=False, L=1000) 

    a = 20
    output[output >= a] = 255
    output[output < a] = 0
    output[output == 255] = mash_num
    cv2.imwrite(os.path.join(OUT_DIR_b1, filename + '.png'), output)


for fn_img in label_pths_png:
    mask = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)  
    filename = os.path.splitext(os.path.basename(fn_img))[0]
    image = cv2.imread(os.path.join(image_paths_tif, filename + '.tif'))
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    mash_num = 11
    mask[mask != mash_num] = 0
    mask[mask == mash_num] = 255
   
    output = refiner.refine(image, mask, fast=False, L=1000) 

    a = 20
    output[output >= a] = 255
    output[output < a] = 0
    output[output == 255] = mash_num
    cv2.imwrite(os.path.join(OUT_DIR_b2, filename + '.png'), output)





