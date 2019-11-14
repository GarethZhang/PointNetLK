import pykitti
import os
import numpy as np
from scipy.stats import binned_statistic
import pickle
import cv2

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def ensure_file_dir_exists(path):
    make_dir_if_not_exist(os.path.dirname(path))
    return path

base_dir = '/home/haowei/Documents/files/research/datasets/KITTI/dataset'
sequences = ["04"]
for seq in sequences:
    data = pykitti.odometry(base_dir, seq, frames=None)