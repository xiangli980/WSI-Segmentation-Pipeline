from skimage import measure
import numpy as np
import matplotlib.pyplot as plt

def dice(y_true, y_pred):
    smooth = 0.2
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def show_output(model,data,dscl,dsch):
    model.set_input(data["img"],data["msk"])
    model.test()
    pred = model.get_pred()
    pred = pred.detach().squeeze().cpu().numpy()
    pred = np.moveaxis(pred,0,-1) 
    result = np.argmax(pred,axis=2)
    mask = data["msk"].squeeze().numpy()
    score = dice(mask,result)
    if(score<dsch and score>dscl):
      contours_p = measure.find_contours(result, 0.8)
      contours_y = measure.find_contours(mask, 0.8)
      img = np.moveaxis(data["img"].squeeze().numpy(),0,-1)
     
      #visualize its result
      fig, ax = plt.subplots(1,5,figsize=(10,10))
      [axi.set_axis_off() for axi in ax.ravel()]

      ax[0].imshow(img)
      ax[0].set_title(data["name"][0][5:-4])
      for n, contour in enumerate(contours_p):
          ax[0].plot(contour[:, 1], contour[:, 0], 'b', linewidth=2)
      for n, contour in enumerate(contours_y):
          ax[0].plot(contour[:, 1], contour[:, 0], '#FFFB33', linewidth=2)
      
      ax[1].imshow(mask,cmap='gray')
      ax[1].set_title('GT')
      ax[2].imshow(result,cmap='gray')
      ax[2].set_title('Pred DSC:{0:.2f}'.format(score))
      ax[3].imshow(pred[:,:,1])
      ax[3].set_title('layer1')
      ax[4].imshow(pred[:,:,0])
      ax[4].set_title('layer0')
      plt.show()

    return pred

def show_hist(phase,dataLoader,model):
    dices = []
    phase = "test"
    # for each img make prediction and compute dice score
    for ii , data in enumerate(dataLoader[phase]): 
        model.set_input(data["img"],data["msk"])
        model.test()
        pred = model.get_pred()
        pred = pred.detach().squeeze().cpu().numpy()
        pred = np.moveaxis(pred,0,-1) 
        result = np.argmax(pred,axis=2)
        mask = data["msk"].squeeze().numpy()
        dices.append(dice(mask,result))
    # plot testing deice score histogram distribution
    mean=np.asarray(dices).mean()
    plt.figure()
    plt.hist(dices,range=(0, 1))
    plt.ylabel('# of patches')
    plt.xlabel('DSC')
    plt.title('mean {0} dice:{1:.3f}'.format(phase,mean))