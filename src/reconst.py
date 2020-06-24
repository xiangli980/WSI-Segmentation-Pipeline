import math
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as skio 
import matplotlib.pyplot as plt
from skimage import measure,morphology,color
from skimage.transform import resize
from scipy import ndimage
from torchvision import transforms

def dice(y_true, y_pred):
    smooth = 0.2
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def pred_output(model, img, msk):
    model.set_input(img,msk)
    model.test()
    pred = model.get_pred()
    pred = pred.detach().squeeze().cpu().numpy()
    pred = np.moveaxis(pred,0,-1) 
    result = np.argmax(pred,axis=2)
    return result

# pad to a valid size for window-slide crop method
def pad_img(im,crop):
    h, w = im.shape[0], im.shape[1]  
    hn=math.ceil(h/crop)
    wn=math.ceil(w/crop)
    delta_w = wn*crop 
    delta_h = hn*crop
    bottom = delta_h-h
    right = delta_w-w
    color = [255, 255, 255] #pad white
    new_im = cv2.copyMakeBorder(im, 0, bottom, 0, right, cv2.BORDER_CONSTANT,value=color)
    new_h,new_w = new_im.shape[0], new_im.shape[1]
    return new_im,hn,wn
def pad_msk(im,crop):
    h, w = im.shape[0], im.shape[1]
    hn=math.ceil(h/crop)
    wn=math.ceil(w/crop)
    delta_w = wn*crop 
    delta_h = hn*crop
    bottom = delta_h-h
    right = delta_w-w
    color = [0, 0, 0]  # pad black
    new_im = cv2.copyMakeBorder(im, 0, bottom, 0, right, cv2.BORDER_CONSTANT,value=color)
    new_h,new_w = new_im.shape[0], new_im.shape[1]
    return new_im

# get input patches with an edge which will be cropped in the future
def ext(io,ii,jj,patchsize,padding):
    images1 = []
    for i in range(ii):
        for j in range(jj):
            pt = []    
            # set patch at current i,j
            if (i==0 and j == 0):
                pt = io[i*patchsize:i*patchsize+patchsize+padding,j*patchsize:j*patchsize+patchsize+padding]
            if (i==0 and j == (jj-1)):
                pt = io[i*patchsize:i*patchsize+patchsize+padding,j*patchsize-padding:j*patchsize-padding+patchsize+padding]
            if (i == (ii-1) and j == (jj-1)):
                pt = io[i*patchsize-padding:i*patchsize-padding+patchsize+padding,j*patchsize-padding:j*patchsize-padding+patchsize+padding]
            if (i == (ii-1) and j==0):
                pt = io[i*patchsize-padding:i*patchsize-padding+patchsize+padding,j*patchsize:j*patchsize+patchsize+padding]
            if (i==0 and j>0 and j<(jj-1)):
                pt = io[i*patchsize:i*patchsize+patchsize+padding,j*patchsize-int(padding/2):j*patchsize-int(padding/2)+patchsize+padding]
            if (i==(ii-1) and j>0 and j<(jj-1)): 
                pt = io[i*patchsize-padding:i*patchsize-padding+patchsize+padding,j*patchsize-int(padding/2):j*patchsize-int(padding/2)+patchsize+padding]
            if (j==0 and i>0 and i<(ii-1)):
                pt = io[i*patchsize-int(padding/2):i*patchsize-int(padding/2)+patchsize+padding,j*patchsize:j*patchsize+patchsize+padding]
            if (j==(jj-1) and i>0 and i<(ii-1)):
                pt = io[i*patchsize-int(padding/2):i*patchsize-int(padding/2)+patchsize+padding,j*patchsize-padding:j*patchsize-padding+patchsize+padding]
            if (i>0 and i<(ii-1) and j>0 and j<(jj-1)): 
                pt = io[i*patchsize-int(padding/2):i*patchsize-int(padding/2)+patchsize+padding,j*patchsize-int(padding/2):j*patchsize-int(padding/2)+patchsize+padding]
            images1.append(pt)
    return images1

# convert to tensor for model prediction
def to_tensor(io_arr_out):
    skip = []
    inputs = []
    trans = transforms.ToTensor()
    # convert input patch to tensor
    for im in io_arr_out:
        im1 = trans(im)
        inputs.append(im1) 
        # for saving time
        # if it is a background patch, skip=1 (not pass to prediction model)
        hsv = color.rgb2hsv(im)
        area = (hsv[:,:,1]>0.1)*1
        area = np.sum(area)
        if area == 0: # whole white background
          skip.append(1)
        else:
          skip.append(0)
    return inputs, skip

def get_results(model,inputs,skip):
    results = []
    for i in range(len(inputs)):
        # (4 dim batch input for model)
        img4d = inputs[i].unsqueeze(0)
        if ~skip[i]:
          out = pred_output(model,img4d,img4d)
        else:
          out = np.zeros((patchsize+padding, patchsize+padding), dtype=bool)
        results.append(out)
    return results

# get center cropped predictions and stitch together 
def reconst_mask(new_mask,results,ii,jj,patchsize,padding):    
    k=0
    for i in range(ii):
        for j in range(jj): 
          if (i==0 and j == 0):
            new_mask[i*patchsize:i*patchsize+patchsize,j*patchsize:j*patchsize+patchsize]=results[k][0:patchsize,0:patchsize]
          if (i==0 and j == (jj-1)):
            new_mask[i*patchsize:i*patchsize+patchsize,j*patchsize:j*patchsize+patchsize]=results[k][0:patchsize,padding:padding+patchsize]   
          if (i == (ii-1) and j == (jj-1)): 
            new_mask[i*patchsize:i*patchsize+patchsize,j*patchsize:j*patchsize+patchsize]=results[k][padding:padding+patchsize,padding:padding+patchsize]
          if (i == (ii-1) and j==0):
            new_mask[i*patchsize:i*patchsize+patchsize,j*patchsize:j*patchsize+patchsize]=results[k][padding:padding+patchsize,0:patchsize]
          if (i==0 and j>0 and j<(jj-1)):
            new_mask[i*patchsize:i*patchsize+patchsize,j*patchsize:j*patchsize+patchsize]=results[k][0:patchsize,int(padding/2):int(padding/2)+patchsize]
          if (i==(ii-1) and j>0 and j<(jj-1)):
            new_mask[i*patchsize:i*patchsize+patchsize,j*patchsize:j*patchsize+patchsize]=results[k][padding:padding+patchsize,int(padding/2):int(padding/2)+patchsize]
          if (j==0 and i>0 and i<(ii-1)):
            new_mask[i*patchsize:i*patchsize+patchsize,j*patchsize:j*patchsize+patchsize]=results[k][int(padding/2):int(padding/2)+patchsize,0:patchsize]
          if (j==(jj-1) and i>0 and i<(ii-1)):
            new_mask[i*patchsize:i*patchsize+patchsize,j*patchsize:j*patchsize+patchsize]=results[k][int(padding/2):int(padding/2)+patchsize,padding:padding+patchsize]
          if (i>0 and i<(ii-1) and j>0 and j<(jj-1)): 
            new_mask[i*patchsize:i*patchsize+patchsize,j*patchsize:j*patchsize+patchsize]=results[k][int(padding/2):int(padding/2)+patchsize,int(padding/2):int(padding/2)+patchsize]
          k += 1
    return new_mask

def get_instance(contour,shape):
    # Create an empty image
    r_mask = np.zeros(shape).astype('bool')
    # Create a contour by using the contour coordinates rounded to their nearest integer value
    r_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
    # Fill in the hole created by the contour boundary
    r_mask = ndimage.binary_fill_holes(r_mask)
    return r_mask
def get_masks(contours,shape):
    mask_y = []
    # for each contour, create a mask image in a whole slide size
    for contour in contours:
        mask = get_instance(contour,shape)
        mask_y.append(mask)
    mask_y=np.asarray(mask_y)
    return mask_y

# remove small artifacts
def rm_small(contours,masks):
  contours_n = []
  masks_n = []
  for i in range(len(masks)):
      if (np.sum(masks[i])>200):
          contours_n.append(contours[i])
          masks_n.append(masks[i])
  return np.asarray(contours_n),np.asarray(masks_n)

# compare prediction to GT gloms
def make_compare(mask_y,mask_p,contours_p,contours_y):
    # check gloms on target 
    flag_y = np.zeros((len(mask_y)))
    flag_p = np.zeros((len(mask_p)))
    # for every ith mask label
    for i in range(len(mask_y)):
        max_dice = 0.1
        for j in range(len(mask_p)):
            # compute dice between ith mask and jth predict  
            d = dice(mask_y[i],mask_p[j])
            for j in range(len(mask_p)):
            # compute dice between ith mask and jth predict
              a,b,c,d = max(contours_p[j][:,0]),min(contours_p[j][:,0]),max(contours_p[j][:,1]),min(contours_p[j][:,1])
              xj = (a+b)/2
              yj = (c+d)/2
              a,b,c,d = max(contours_y[i][:,0]),min(contours_y[i][:,0]),max(contours_y[i][:,1]),min(contours_y[i][:,1])
              xi = (a+b)/2
              yi = (c+d)/2
              if (xj-xi)*(xj-xi)+(yj-yi)*(yj-yi) < 1600:  # restrict to neighbor gloms, save time 
                d = dice(mask_y[i],mask_p[j])
                if d > max_dice:    
                    flag_y[i] = 1
                    flag_p[j] = 1
    return flag_p,flag_y

def save_plot(flag_p,flag_y,contours_p,contours_y,dice, msk1,img_id, img_type, ii, jj, patchsize):
    fig, ax = plt.subplots()
    tol = flag_y.shape[0]
    cntp = flag_p.shape[0]
    find = 0.
    for n, contour in enumerate(contours_p):
        if flag_p[n]==1:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.2, color = 'b')
        else:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.2, color = 'g')

    for n, contour in enumerate(contours_y):
        if flag_y[n] ==0:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.2, color = 'y')
        else: 
            find += 1
    
    precise = find/(cntp+0.00001)*100
    recall = find/(tol+0.00001)*100
    f1 = (2*precise*recall/(precise+recall+0.00001))*0.01
    ax.imshow(msk1,cmap='gray')        
    ax.set_title(' precision {0:.2f}%[{1}/{2}] recall {3:.2f}%[{4}/{5}] F1 {6:.2f} DSC {7:.2f}'.format(precise, 
                                int(find), cntp, recall, int(find), tol, f1, dice))

    plt.savefig('./log/reconst/{}_{}.png'.format(img_id,img_type),dpi=1000)
    plt.show()
    return precise, cntp, recall, int(find), tol, f1, dice