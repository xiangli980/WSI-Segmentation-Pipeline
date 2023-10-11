# WSI-Segmentation-Pipeline
This is a repo for Deep Learning Segmentation of Glomeruli in WSI(Whole Slide Image) of H&E frozen Donor Kidney Biopsy.
## Requirements
Python 3.9.18 <br>
PyTorch 2.0.1<br>
torchvision 0.15.2<br>
openslide-python 1.3.1 (for WSI preprocess)<br> 
```
conda create --name myenv python=3.9
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge openslide
conda install -c conda-forge openslide-python
pip install -r requirement.txt
```

## WSI Preprocess
Whole Slide Images are  high resolution images which are usually stored in pyramidal tiled (.svs) files. We first convert them into downscaled .PNG format to be used in the following steps.<br>
- Input `.svs` file into `\WSI_Preprocess\slides`
- Run `step1_read_WSI` to fetch whole slide region into `\data\WSI` and extract image patches for training and testing in `\WSI_Preprocess\masks_patch`, `\WSI_Preprocess\slides_patch`. Notice that whole slide level prediction of non-sclerotic glom require 2.5X magnification, while sclerotic glom prediction require 5X magnification, in the pretrained models. Please downsample WSI accordingly if using the pretrained models.    
- Run `step2_make_HDF5` to organize patches into training/testing cohorts of HDF5 file (.pytable) into `\data`  

## Train
Defualt segmentation model is a Unet, requiring input image size of 256x256x3 and mask size of 256x256  
- Input `{dataname}_{phase}.pytable` into `\data`
- Run `train_demo` to train the model, which containing some training parameters that can be adjusted
- Save model weights `{model}_{dataname}_{epoch}.pth` to `\log` 
## Test
- Input model weights from `\log` and testing patches from `\data`
- Run `test_demo` to show testing data predictions and model performance 
## Reconstruct
To make prediction on the whole slide, we need to divide the slide into patches and stitch predicted results together to generate whole prediction mask 
- Input WSI from `\data\WSI`
- Run `reconstruct_demo` to perform center-crop window-slide prediction and stitching strategy 
- Generate whole slide prediction and performance data into `\log\reconst`

## Reference
For papers:<br>
Xiang Li et al. Deep learning segmentation of glomeruli on kidney donor frozen sections. J Med Imaging 2021 <br>
Aritra Ray et al. Decoding the Encoder. SoutheastCon 2023  
   
