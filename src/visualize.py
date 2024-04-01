import os
import glob

import cv2
import numpy as np
from PIL import Image


class_rgb = {
    "bg": [0, 0, 0],
    "tree": [71, 148, 101],
    "rangeland": [101, 251, 72],
    "bareland": [140, 0, 6],
    "agric land type 1": [104, 186, 93],
    "road type 1": [246, 246, 246],
    "sea, lake, & pond": [0, 109, 203],
    "building type 1": [250, 151, 101],
    "road type 2": [175, 176, 179],
    "river": [0, 82, 250],
    "boat & ship": [255, 240, 75],
    "agric land type 2": [194, 249, 75],
}

class_gray = {
    "bg": 0,
    "tree": 1,
    "rangeland": 2,
    "bareland": 3,
    "agric land type 1": 4,
    "road type 1": 5,
    "sea, lake, & pond": 6,
    "building type 1": 7,
    "road type 2": 8,
    "river": 9,
    "boat & ship": 10,
    "agric land type 2": 11,
}


def label2rgb(a):
    """
    a: labels (HxW)
    """
    out = np.zeros(shape=a.shape + (3,), dtype="uint8")
    for k, v in class_gray.items():
        out[a == v, 0] = class_rgb[k][0]
        out[a == v, 1] = class_rgb[k][1]
        out[a == v, 2] = class_rgb[k][2]
    return out


if __name__ == "__main__":
    src_dir = "results/preds"
    dst_dir = "results/preds_vis"
    os.makedirs(dst_dir, exist_ok=True)
    pths = glob.glob(src_dir + "/*.png")
    for img_pth in pths:
        img = cv2.imread(img_pth, -1)
        rgb_img = label2rgb(img)
        Image.fromarray(rgb_img).save(os.path.join(dst_dir, os.path.basename(img_pth)))
