import numpy as np
import sys
import json
from tiffslide import TiffSlide
import girder_client
from utils.downsample_image import downsample_image

from src.makemodel import GlomNet
from src.reconst import pad_img, get_results, reconst_mask, to_tensor, ext

sys.path.append("..")
from dsacode.utils.annotation_converter import converter, get_contour_points

PATCHSIZE = 256 # center area size
PADDING = 64

def predict(args):

    isTrain = False  # set model in testing mode
    isContinue = False
    savedir = "./log/"
    loadpath = args.modelfile

    model = GlomNet(isTrain, isContinue, savedir, loadpath, "UNet")

    slide = TiffSlide(args.input_file)
    img = downsample_image(slide)

    #msk = cv2.imread("./data/WSI/{}".format(msk_dir))/255
    print("slide size", img.shape)

    # pad to a suitable size for window-slide patch extraction
    #msk = pad_msk(msk,patchsize)[:,:,0]
    img,rr,cc = pad_img(img, PATCHSIZE)
    # extract all patches from the whole slide with calculated rows and columns
    io_arr_out = np.array(ext(img,rr,cc,PATCHSIZE,PADDING))
    io_arr_out = io_arr_out.reshape(-1,PATCHSIZE+PADDING, PATCHSIZE+PADDING,3)
    print("slide size after padding", img.shape)
    print("collected patches size",io_arr_out.shape)
    
    # convert patches into tensor format and add skip flag to 
    # patches on white backgrounds for computation saving
    inputs, skip = to_tensor(io_arr_out)
    results = get_results(model, inputs, skip)
    
    # reconstruct whole slide prediction by stitching output results
    newmask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    new_mask = reconst_mask(newmask,results,rr,cc,PATCHSIZE,PADDING)
    new_mask = new_mask.astype('bool')

    wsiMask = new_mask*1
    points = (get_contour_points(wsiMask.astype(np.uint8), 1,1, offset={'X': 0,'Y': 0}))

    folder = args.base_dir
    girder_folder_id = folder.split('/')[-2]    
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken) 
    file_name = args.input_file.split('/')[-1]
    files = list(gc.listItem(girder_folder_id))
    item_dict = dict()
    for file in files:
        d = {file['name']: file['_id']}
        item_dict.update(d)
    print(item_dict)
    print(item_dict[file_name])
    #gc.uploadFileToItem(item_dict[file_name], f'{path_stem}_mask.json', reference=None, mimeType=None, filename="Annotation.json", progressCallback=None)
    _ = gc.post(path='annotation',parameters={'itemId':item_dict[file_name]}, data = json.dumps(converter(points,['gloms'])[0]))
    print("done")
