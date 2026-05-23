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
        def __init__(self, out_nc=1):
            super().__init__()
            
            self.act = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(2,2)
            
            self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            
            self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            
            self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            
            self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  
                
            
            self.flat = nn.Flatten()
            
            self.fl1 = nn.Linear(7*7*512, 256)
            self.fl2 = nn.Linear(256, 1)

            
            
        def forward(self, x):
            out = self.conv1_1(x)        
            out = self.act(out)
            out = self.conv1_2(out)
            out = self.act(out)
         
            out = self.maxpool(out)
            
            out = self.conv2_1(out)        
            out = self.act(out)
            out = self.conv2_2(out)
            out = self.act(out)

            out = self.maxpool(out)
            
            out = self.conv3_1(out)        
            out = self.act(out)
            out = self.conv3_2(out)
            out = self.act(out)
            out = self.conv3_3(out)
            out = self.act(out)
            
            out = self.maxpool(out)
        
            out = self.conv4_1(out)        
            out = self.act(out)
            out = self.conv4_2(out)
            out = self.act(out)
            out = self.conv4_3(out)
            out = self.act(out)
            
            out = self.maxpool(out)
            

            out = self.flat(out)
            
            out = self.fl1(out)
            out = self.act(out)
            out = self.fl2(out)

            

            
            return out
        
        return
