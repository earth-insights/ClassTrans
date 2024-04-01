import os
import os.path as osp
import cv2


def slide_window(inputs, stride=(128, 128), crop_size=(128, 128), filename='', dst_dir=''):
    """
    filename: xxx.xx
    """
    
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    h_img, w_img, bands = inputs.shape
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    id_idx = 0
    filename, postfix = filename.split('.')
    postfix = 'png'
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = inputs[y1:y2, x1:x2, :]

            cv2.imwrite(osp.join(dst_dir, filename+f'_{str(id_idx).zfill(6)}.'+postfix), crop_img)
            id_idx += 1
                

if __name__ == '__main__':
    img_dir = 'data/cvpr2024_oem_ori'
    dst_dir = 'data/cvpr2024_oem_crop-128'
    os.makedirs(dst_dir, exist_ok=True)

    for img_name in os.listdir(img_dir):
        img_path = osp.join(img_dir, img_name)
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        slide_window(img, stride=(128, 128), crop_size=(256, 256), filename=filename, dst_dir=dst_dir)