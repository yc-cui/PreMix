import scipy.misc as misc
from scipy import signal
from scipy import ndimage
import numpy as np
import cv2
import os
import torch


def check_and_make(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def regularize_inputs(*args):
    output = []
    for v in args:
        output.append(torch.clip(v, 0., 1.))
    return tuple(output)
