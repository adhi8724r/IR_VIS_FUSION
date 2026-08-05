import torch
import numpy as np
import cv2

def tensor_to_uint8(tensor: torch.tensor):
    img = tensor.clamp(0,1) * 255
    return img.squeeze().cpu().numpy().astype(np.uint8)

def pad_to_128(img):
    h, w = img.shape[:2]
    pad_h = (128 - h % 128) % 128
    pad_w = (128 - w % 128) % 128

    img_pad = cv2.copyMakeBorder(
        img,
        0, pad_h,
        0, pad_w,
        cv2.BORDER_CONSTANT,
        value=128
    )

    print(f'image pad = {img_pad.shape}')
    return img_pad, h, w


def unpad(img_pad, h, w):
    return img_pad[:h, :w]