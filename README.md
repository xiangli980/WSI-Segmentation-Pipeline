# WSI-Segmentation-Pipeline
This is a repo for Deep Learning Segmentation of Glomeruli in WSI(Whole Slide Image) of Donor Kidney Biopsy.
## Requirements
Python 3.7.4 <br>
PyTorch 1.4.0<br>
torchvision 0.5.0<br>
openslide-python 1.1.1 (for WSI preprocess)<br> 
skimage<br>
sklearn<br>
cv2<br>
matplotlib<br>

## WSI Preprocess
Whole Slide Images are  high resolution images which are usually stored in pyramidal tiled (.svs) files. We first convert them into downscaled .PNG format to be used in the following steps.<br>
- Input `.svs` file to `\WSI_Preprocess\slides`
- Run `step1_read_WSI` to generate image patch data in `\WSI_Preprocess\masks_patch`,`\WSI_Preprocess\slides_patch` and whole slide in `\data`
- Run `step2_make_HDF5` to organize patches into training/testing cohorts of HDF5 file (.pytable) in `\data`  

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
- Run `reconstruct_demo` to perform center-crop window-slide stitch strategy 
- Generate whole slide prediction and performance data into `\log\WSI`


   