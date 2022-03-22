import numpy as np
from PIL import Image
import io
import cv2
import torchvision.transforms as transforms
import torch
from icecream import ic

class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff).convert('RGB')

def rescale_bbox(x_scale, y_scale, bbox, size):
    x = int(np.round(bbox[0] * x_scale))
    if x > size[0]-1:
        x = 0
    y = int(np.round(bbox[1] * y_scale))
    if y > size[1]-1:
        y = 0
    w = int(np.round(bbox[2] * x_scale))
    if w > size[0]:
        w = size[0]-1
    h = int(np.round(bbox[3] * y_scale))
    if h > size[1]:
        h = size[1]-1
    new_bbox = [x, y, w, h]
    return new_bbox

def get_indices(bbox, offset, size):
    x, y, w, h = bbox
    y1 = max(0, y - offset)
    y2 = min(size[1]-1, y + h + 2 * offset-1)
    x1 = max(0, x - offset)
    x2 = min(size[0]-1,  x+ w + 2 * offset-1)

    if (x2-x1) < 1:
        x1 = 0
        x2 = size[0] - 1

    if (y2-y1) < 1:
        y1 = 0
        y2 = size[1] - 1

    return x1, x2, y1, y2

def combine_bbox(bboxes):
    '''
    bboxes: Nx4, xywh
    '''
    l = bboxes[:,0].min()
    u = bboxes[:,1].min()
    r = (bboxes[:,0] + bboxes[:,2]).max()
    b = (bboxes[:,1] + bboxes[:,2]).max()
    w = r - l
    h = b - u
    return np.array([l, u, w, h])

def mask_to_bbox(mask):
    mask = (mask == 1)
    if np.all(~mask):
        return [0, 0, 0, 0]
    assert len(mask.shape) == 2
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin.item(), rmin.item(), cmax.item() + 1 - cmin.item(), rmax.item() + 1 - rmin.item()] # xywh

def bbox_iou(b1, b2):
    '''
    b: (x1,y1,x2,y2)
    '''
    lx = max(b1[0], b2[0])
    rx = min(b1[2], b2[2])
    uy = max(b1[1], b2[1])
    dy = min(b1[3], b2[3])
    if rx <= lx or dy <= uy:
        return 0.
    else:
        interArea = (rx-lx)*(dy-uy)
        a1 = float((b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = float((b2[2] - b2[0]) * (b2[3] - b2[1]))
        return interArea / (a1 + a2 - interArea)

def crop_padding(img, roi, pad_value):
    '''
    img: HxW or HxWxC np.ndarray
    roi: (x,y,w,h)
    pad_value: [b,g,r]
    '''
    need_squeeze = False
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
        need_squeeze = True
    assert len(pad_value) == img.shape[2]
    x,y,w,h = roi
    x,y,w,h = int(x),int(y),int(w),int(h)
    H, W = img.shape[:2]
    output = np.tile(np.array(pad_value), (h, w, 1)).astype(img.dtype)
    if bbox_iou((x,y,x+w,y+h), (0,0,W,H)) > 0:
        output[max(-y,0):min(H-y,h), max(-x,0):min(W-x,w), :] = img[max(y,0):min(y+h,H), max(x,0):min(x+w,W), :]
    if need_squeeze:
        output = np.squeeze(output)
    return output


####################################

def get_affine_transform(center, scale, rot, output_size):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale * 200.0

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.])
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

