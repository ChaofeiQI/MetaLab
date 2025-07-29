# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2025
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Bernoulli
import cv2, os                                                          
from joblib import Parallel, delayed                                    
from pathos.multiprocessing import ProcessingPool as Pool               
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor 
from colorama import init, Fore
init()  # Init Colorama

def Tensor_rgb_to_lab(image):
    # non negative
    image = torch.clamp(image, 0, None)
    image = torch.where(image > 0.0031308, 
                        1.055 * (image ** (1/2.4)) - 0.055, 12.92 * image)
    image *= 100.0

    # RGB to XYZ
    mat = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]], device=image.device)    
    xyz = torch.matmul(image, mat.T)
    xyz = torch.clamp(xyz, min=0)
    
    # XYZ to LAB
    xyz_ref = torch.tensor([95.047, 100.000, 108.883], device=image.device)
    xyz /= xyz_ref
    xyz = torch.where(xyz > 0.008856, xyz ** (1/3), (xyz * 7.787) + (16/116))
    l_c = 116 * xyz[..., 1] - 16
    a_c = 500 * (xyz[..., 0] - xyz[..., 1])
    b_c = 200 * (xyz[..., 1] - xyz[..., 2])
    
    # Normalize to [-1, 1]
    l_c = (l_c / 100) * 2 - 1
    a_c = (a_c + 88) / 176 * 2 - 1
    b_c = (b_c + 107) / 214 * 2 - 1
    
    # channels stack    
    lab = torch.stack([l_c, l_c, a_c, b_c], dim=-1)
    return lab

def LAB_Space_Transfer_Batch(image_batch, mode='Tensor'):
    image_batch_rgb = (image_batch + 3) / 6  # 将 [-3, 3] 转换为 [0, 1]
    # We offer a variety of strategies
    if mode == 'Tensor':
        with torch.no_grad():
            image_batch_lab = Tensor_rgb_to_lab(image_batch_rgb.permute(0, 2, 3, 1))
            return image_batch_lab.permute(0, 3, 1, 2).to(image_batch.device)
    
    elif mode == 'OpenCV':
        import cv2 
        with torch.no_grad():
            b, h, w, c = image_batch_rgb.shape
            image_batch_lab = np.empty((b, h, w, 4), dtype=np.float32)
            for i in range(b):
                image = (image_batch_rgb[i].numpy() * 255).astype(np.uint8)  # 转换为 [0, 255]
                image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
                image_lab = image_lab.astype(np.float32)
                image_batch_lab[i] = np.stack([
                    (image_lab[:,:,0] / 100) * 2 - 1,  # L
                    (image_lab[:,:,0] / 100) * 2 - 1,  # L
                    (image_lab[:,:,1] + 88) / 176 * 2 - 1,  # a
                    (image_lab[:,:,2] + 107) / 214 * 2 - 1,  # b
                ], axis=-1)
            image_batch_lab_tensor = torch.tensor(image_batch_lab, dtype=torch.float32).permute(0, 3, 1, 2)
            return image_batch_lab_tensor.to(image_batch.device)
      
    elif mode in ['concurrent-MT', 'concurrent-MP', 'joblib', 'pathos']:
        with torch.no_grad():
            image_batch_rgb = image_batch.cpu().permute(0, 2, 3, 1).numpy()
            b, h, w, c = image_batch_rgb.shape
            image_batch_lab = np.empty((b, h, w, 4), dtype=np.float32)
            if mode == 'concurrent-MT':
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(Tensor_rgb_to_lab, image_batch_rgb))
            elif mode == 'concurrent-MP':
                with ProcessPoolExecutor() as executor:
                    results = list(executor.map(Tensor_rgb_to_lab, image_batch_rgb))
            elif mode == 'joblib':
                results = Parallel(n_jobs=-1)(delayed(Tensor_rgb_to_lab)(image) for image in image_batch_rgb)
            elif mode == 'pathos':
                with Pool() as pool:
                    results = pool.map(Tensor_rgb_to_lab, image_batch_rgb)
            for i, image_lab in enumerate(results):
                image_batch_lab[i] = image_lab
            image_batch_lab_tensor = torch.tensor(image_batch_lab, dtype=torch.float32).permute(0, 3, 1, 2)
            return image_batch_lab_tensor.to(image_batch.device)

    raise ValueError("Unsupported mode: {}".format(mode))


####################
# 搭建LabNet网络结构
####################
class LabNet(nn.Module):
    """ LabNet Backbone"""
    def __init__(self, emb_size, encoder_flag=False):
        super(LabNet, self).__init__()
        self.hidden = 96
        self.emb_size = emb_size
        self.last_hidden = self.hidden * 25  if not encoder_flag else self.hidden
        # 1)LAB_Block-1：
        self.LAB_Block_1 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=self.hidden, kernel_size=3, padding=1, bias=False, groups=2),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # 2)LAB_Block-2：
        self.LAB_Block_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden, out_channels=int(self.hidden*1.5), kernel_size=3, bias=False, groups=2),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # 3)LAB_Block-3：
        self.LAB_Block_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5), out_channels=self.hidden*4, kernel_size=3, padding=1, bias=False, groups=2),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        # Penultimate Embedding
        self.downsamp = nn.MaxPool2d(kernel_size=2)
        self.embd_second_L = nn.Sequential(nn.Linear(in_features=self.last_hidden*2, out_features=self.emb_size, bias=True),
                                           nn.BatchNorm1d(self.emb_size))
        self.embd_second_ab= nn.Sequential(nn.Linear(in_features=self.last_hidden*2, out_features=self.emb_size, bias=True),
                                           nn.BatchNorm1d(self.emb_size))
        # 4)LAB_Block-4：
        self.LAB_Block_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*4, out_channels=self.hidden*4, kernel_size=3, padding=1, bias=False, groups=2),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        # Last-layer Embedding
        self.embd_last_L  = nn.Sequential(nn.Linear(in_features=self.last_hidden*2, out_features=self.emb_size, bias=True),
                                          nn.BatchNorm1d(self.emb_size))
        self.embd_last_ab = nn.Sequential(nn.Linear(in_features=self.last_hidden*2, out_features=self.emb_size, bias=True),
                                          nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        feat_embd = []                                           
                
        # LAB Space Transformation
        LAB = LAB_Space_Transfer_Batch(input_data, 'Tensor').to(input_data.device)     
        # LAB = LAB_Space_Transfer_Batch(input_data.cuda(), 'OpenCV').cuda()        
        # LAB = LAB_Space_Transfer_Batch(input_data.cuda(), 'joblib').cuda()       
        # LAB = LAB_Space_Transfer_Batch(input_data.cuda(), 'pathos').cuda()       
        # LAB = LAB_Space_Transfer_Batch(input_data.cuda(), 'concurrent-MP').cuda()
        # LAB = LAB_Space_Transfer_Batch(input_data.cuda(), 'concurrent-MT').cuda()

        # Grouped Feature Extraction
        feat_1 = self.LAB_Block_1(LAB)    
        feat_2 = self.LAB_Block_2(feat_1)     
        feat_3 = self.LAB_Block_3(feat_2)
        output = self.LAB_Block_4(feat_3)

        # Penultimate Embedding
        output_data0_L = self.downsamp(feat_3[:,0:int(feat_3.shape[1]//2),:,:]) 
        output_data0_ab = self.downsamp(feat_3[:,int(feat_3.shape[1]//2):,:,:]) 
        feat_second_L=self.embd_second_L(output_data0_L.reshape(output_data0_L.size(0), -1))    
        feat_second_ab=self.embd_second_ab(output_data0_ab.reshape(output_data0_ab.size(0), -1))
        feat_second = torch.cat((feat_second_L, feat_second_ab), dim=1) 
        feat_embd.append(feat_second)

        # Last-layer Embedding
        feat_last_L=self.embd_last_L(output[:,0:int(output.shape[1]//2),:,:].reshape(output.size(0), -1))
        feat_last_ab=self.embd_last_ab(output[:,int(output.shape[1]//2):,:,:].reshape(output.size(0), -1))
        feat_last = torch.cat((feat_last_L, feat_last_ab), dim=1) 
        feat_embd.append(feat_last)
        
        return feat_embd


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dummy_input = torch.randn(64, 3, 84, 84)  
    t1=time.time()
    
    emb_size = 128 
    model = LabNet(emb_size).cuda()
    print('model:',model)
    t2=time.time()

    for i in range(10):
        outputs = model(dummy_input.cuda())
    t3=time.time()

    print('LabNet Instantiation(s):{:.6f}'.format(t2-t1)) 
    print('Testing Time Consuming(s):{:.6f}'.format(t3-t2))