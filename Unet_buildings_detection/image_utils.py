import numpy as np
from osgeo import gdal,osr,ogr
import torch
from scipy.ndimage.morphology import distance_transform_bf



def standardize(data):
    '''
    Standardize the input data of the network
    :param data to be standardized (size size_batch x WIDTH x HEIGHT x number of channels) 
    
    returns data standardized size size_batch x WIDTH x HEIGHT x number of channels 
    
    '''

    WIDTH=data.shape[1]
    HEIGHT=data.shape[2]
    channels=data.shape[3]
    
    
    mean_t=torch.mean(data.view(len(data)*WIDTH*HEIGHT,channels),0)
    std_t=torch.std(data.view(len(data)*WIDTH*HEIGHT,channels), 0)
    data=(data-mean_t)/std_t

    #For normalization 
    min_t=torch.min(data.view(len(data)*WIDTH*HEIGHT,channels), 0)
    max_t=torch.max(data.view(len(data)*WIDTH*HEIGHT,channels), 0)
    data=(data-min_t[0])/((max_t[0]-min_t[0]))

    return data


def distance_map_batch(Y,threshold=20,bins=15):
    
    """
    Compute the distance map following https://arxiv.org/pdf/1709.05932.pdf 
    Y: One hot pixel wise labels ( size_batch x WIDTH x HEIGHT x nb_classes) with nb_classes=2 
               Y[:,:,:,0] should represent the background mask
               Y[:,:,:,1] should represent the buildings mask
    threshold: distance threshold
    bins: number of bins considered for the distance map
    Default values are computed for resolution of input image of 50cm per pixel
    return torch distance map for buildings (size_batch x WIDTH x HEIGHT x bins)
    """
    Y_dist=[]
    for i in range(len(Y)):
        distance=distance_transform_bf(np.asarray(Y)[i,:,:,1],sampling=2)
        distance=np.minimum(distance,threshold*(distance>0))*(bins-1)/threshold
        inp=torch.LongTensor(distance)
        inp_ = torch.unsqueeze(inp, len(distance.shape))
        one_hot = torch.FloatTensor(distance.shape[0],distance.shape[1], bins).zero_()
        one_hot.scatter_(len(distance.shape), inp_, 1)
        one_hot=np.asarray(one_hot)
        Y_dist.append(one_hot)

    return torch.FloatTensor(np.asarray(Y_dist))
    
    

