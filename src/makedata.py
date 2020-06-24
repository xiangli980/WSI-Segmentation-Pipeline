import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import tables
import cv2
import random,sys

class Dataset(object):
    def __init__(self, fname ,img_transform=None, mask_transform = None):
        
        self.fname=fname
        self.img_transform=img_transform
        self.mask_transform = mask_transform

        self.tables=tables.open_file(self.fname)
        self.nitems=self.tables.root.img.shape[0]
        self.tables.close()
        
        self.img = None
        self.mask = None
      
    def __getitem__(self, index):

        with tables.open_file(self.fname,'r') as db:
            self.img=db.root.img
            self.mask=db.root.msk
            self.pname=db.root.filename
            #get the requested image and mask from the pytable
            img = self.img[index,:,:,:]
            mask = self.mask[index,:,:]
            name = self.pname[index].decode('utf-8')
        
        #in order to use the transformations given by torchvision
        mask = mask[:,:,None].repeat(3,axis=2) 
        
        #get a random seed so that we can reproducibly do the transofrmations
        seed = random.randrange(sys.maxsize) 
        if self.img_transform is not None:
            random.seed(seed)
            img_new = self.img_transform(img)

        if self.mask_transform is not None:
            random.seed(seed)
            mask_new = self.mask_transform(mask)
            mask_new = np.asarray(mask_new)[0,:,:].squeeze()

        return {"img":img_new, "msk":mask_new, "name":name}

    def __len__(self):
        return self.nitems

class Data_Loader():
    
  def __init__(self, dataname, phase, batch, shuffle):
    # set data augmentation methods
    self.shuffle = shuffle

    if phase == "test":
      img_transform = transforms.Compose([
          transforms.ToPILImage(),
          transforms.ToTensor(),
          ])
      mask_transform = transforms.Compose([
          transforms.ToPILImage(),
          transforms.ToTensor(),
          ])
      
    else:
      img_transform = transforms.Compose([
          transforms.ToPILImage(),
          transforms.RandomVerticalFlip(),
          transforms.RandomHorizontalFlip(),
          transforms.RandomAffine(30, translate=None, scale=(0.7,1.3), shear=20, resample=False, fillcolor=0),
          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
          transforms.ToTensor(),
          ])
      mask_transform = transforms.Compose([
          transforms.ToPILImage(),
          transforms.RandomVerticalFlip(),
          transforms.RandomHorizontalFlip(),
          transforms.RandomAffine(30, translate=None, scale=(0.7,1.3), shear=20, resample=False, fillcolor=0),
          transforms.ToTensor(),
          ])
    
    self.dataset = Dataset(f"./data/{dataname}_{phase}.pytable", 
        img_transform=img_transform , mask_transform = mask_transform)

    self.dataloader = DataLoader(self.dataset, batch_size = batch,
        shuffle = self.shuffle, num_workers = 0)

  def load_data(self):
    return self.dataset

  def vis_data(self):
    for i in range(3):
      ind = random.randint(0, len(self.dataset))
      data = self.dataset[ind]
      fig, ax = plt.subplots(1,2, figsize=(4,4))  # 1 row, 2 columns
      ax[0].imshow(np.moveaxis(np.array(data["img"]),0,-1))
      title = data["name"].split("_")
      ax[0].set_title(title[1]+"_"+title[2])
      ax[1].imshow(data["msk"],cmap="gray")
      ax[0].axis('off')
      ax[1].axis('off')

  def __len__(self):
    return len(self.dataset)

  def __iter__(self):
    for i, data in enumerate(self.dataloader):
      yield data