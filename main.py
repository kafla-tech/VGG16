from operator import index

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import numpy as np
import os
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forvard(self, x):
        
        return