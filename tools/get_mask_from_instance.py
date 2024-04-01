import os
import os.path as osp
import tqdm
import json
import numpy as np
import pycocotools.mask as maskUtils


src_dir = 'output/cvpr2024_oem_crop-256-128_thres-0.1_car'
dst_dir = 'output/cvpr2024_oem_crop-256-128_thres-0.1_car_instance'
os.makedirs(dst_dir, exist_ok=True)

for file_name in tqdm.tqdm(os.listdir(src_dir)):
    if file_name.split('.')[-1] == 'json':

        annotation_file_A = osp.join(src_dir, file_name)

        annotation_file_npy = osp.join(src_dir, file_name.split('.')[0] + '.npy')
        label_npy = np.load(annotation_file_npy)

        iscrowd = [0]
        seg_mask = np.zeros((label_npy.shape[1], label_npy.shape[2])).astype(bool)

        with open(annotation_file_A, 'r') as f:
            dataset = json.load(f)

        g = [g['segmentation'] for g in dataset if g['category_id'] in [0]]

        if len(g) == 0 :
            # cv2.imwrite(osp.join(dst_dir, file_name.split('.')[0] + '.png'), seg_mask.astype('uint8'))
            np.save(osp.join(dst_dir, file_name.split('.')[0] + '.npy'), seg_mask.astype('uint8'))
            continue

        for ins in g:
            area = maskUtils.area(ins)
            # for car
            if 'car' in src_dir:
                if area > 300 or area < 30:
                    continue
            m = maskUtils.decode(ins).astype(bool)
            seg_mask = (seg_mask | m)

        # cv2.imwrite(osp.join(dst_dir, file_name.split('.')[0] + '.png'), seg_mask.astype('uint8'))
        np.save(osp.join(dst_dir, file_name.split('.')[0] + '.npy'), seg_mask.astype('uint8'))