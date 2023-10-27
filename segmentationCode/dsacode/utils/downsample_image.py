import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm 

def downsample_image(slide):
    
    dim_x, dim_y = slide.dimensions
    region_size=2048
    step = 2048
    index_y=np.array(range(0,dim_y+0,step))
    index_x=np.array(range(0,dim_x+0,step))
    imgDown = np.zeros([dim_y//16, dim_x//16,3], dtype='uint8')
    for i,j in tqdm(coordinate_pairs(index_y,index_x)):
        yEnd = min(dim_y+0,i+region_size)
        xEnd = min(dim_x+0,j+region_size)
        xLen=xEnd-j
        yLen=yEnd-i

        dxS=j
        dyS=i
        dxE=j+xLen
        dyE=i+yLen
        im=np.array(slide.read_region((dxS,dyS),0,(xLen,yLen)))[:,:,:3]
        imgpart = zoom(im,(0.0625,0.0625,1), order=0,mode='nearest')
        d1,d2,_ = imgpart.shape
        d1_n,d2_n,_= imgDown[:,:,:1][dyS//16:dyS//16 + d1,dxS//16:dxS//16+d2].shape
        imgDown[:,:,:1][dyS//16:dyS//16 + d1,dxS//16:dxS//16+d2]=imgpart[:d1_n,:d2_n,:1]
        imgDown[:,:,1:2][dyS//16:dyS//16 + d1,dxS//16:dxS//16+d2]=imgpart[:d1_n,:d2_n,1:2]
        imgDown[:,:,2:3][dyS//16:dyS//16 + d1,dxS//16:dxS//16+d2]=imgpart[:d1_n,:d2_n,2:3]
        
    return imgDown

def coordinate_pairs(v1,v2):
    for i in v1:
        for j in v2:
            yield i,j
