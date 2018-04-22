import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt



COLORS = ['#F44336',"#E91E63",'#9C27B0','#673AB7','#3F51B5','#2196F3','#03A9F4','#00BCD4','#4CAF50',
 '#8BC34A','#CDDC39','#FFEB3B','#FFC107','#FF9800','#FF5722']

COLORS = ['#F44336',"#E91E63",'#9C27B0','#673AB7','#3F51B5','#2196F3','#03A9F4','#00BCD4','#4CAF50',
 '#8BC34A','#CDDC39','#FFEB3B','#FFC107','#FF9800','#FF5722']


r_min = 12
r_max = 36
line_width_min = 2
line_width_max = 4
background_intensity = 30.0 / 255.0
def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))

def DrawRandomSquare(img,segments,r_min,r_max,alpha):
    color = hex2rgb( np.random.choice(COLORS) )
    t = np.random.random()
    r = int(t * r_min + (1-t) * r_max)
    i = int(np.random.random()*img.shape[0])
    j = int(np.random.random()*img.shape[1])
    theta = np.pi * np.random.random()
    ri = r*np.cos(theta)
    rj = r*np.sin(theta) 
    pts = [(ri,rj),(-rj,ri),(-ri,-rj),(rj,-ri) ]
    pts = [(i+y,j+x) for (y,x) in pts]
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    canvas = img.copy()
    cv2.fillPoly(canvas,[pts],color)
    cv2.fillPoly(segments,[pts],(0,1,0))
    img = cv2.addWeighted(img, 1.0 - alpha, canvas, alpha, 0, img )
    box = [min(pts[:,:,0])[0],min(pts[:,:,1])[0], max(pts[:,:,0])[0]-min(pts[:,:,0])[0], max(pts[:,:,1])[0] - min(pts[:,:,1])[0] ]
    return img,segments,box

def generateSegmentation(canvas_size, n_max, alpha = 0.5):
    canvas = background_intensity * np.ones((canvas_size,canvas_size,3))
    segments = np.zeros((canvas_size,canvas_size,2))

    for _ in range(np.random.choice(range(n_max))):
        canvas,segments,b = DrawRandomSquare(canvas,segments,r_min,r_max,alpha)
    return canvas,segments



class SimpleSegmentationDataset(Dataset):
    """A simple dataset for image segmentation purpose"""
    def __init__(self, patch_size, n_max, alpha =1.0,virtual_size=1000):
        self.virtual_size = virtual_size
        self.patch_size = patch_size
        self.n_max = n_max
        self.alpha = alpha


    def __len__(self):
        return self.virtual_size

    def __getitem__(self,idx):
        x,y= generateSegmentation(self.patch_size, self.n_max, self.alpha)
        sample = {'input': x, 'groundtruth': y}
        return sample