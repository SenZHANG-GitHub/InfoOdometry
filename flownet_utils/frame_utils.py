import pdb
import numpy as np
from os.path import *
from scipy.misc import imread
from . import flow_utils 
import random

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)
        # if im.shape[2] > 3:
        #     return im[:,:,:3]
        # else:
        #     return im
        return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    return []

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        if len(img.shape) == 2:
            return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw)]
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        if len(img.shape) == 2:
            return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2]
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]
