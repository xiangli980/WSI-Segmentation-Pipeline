import os
from src.models import *

class GlomNet():
  
  def __init__(self, isTrain, isContinue, savedir, loadpath = "", modelname = "UNet"):
    self.gpuid = 0   # default to GPU id 0
    self.isTrain = isTrain
    self.isContinue = isContinue
    self.save_dir = savedir
    self.model_name = modelname
    self.lr = 1e-3
    self.gamma = 2 
    if(torch.cuda.is_available()):
      print(torch.cuda.get_device_properties(self.gpuid))
      torch.cuda.set_device(self.gpuid)
      torch.backends.cudnn.benchmark = True
      self.device = torch.device(f'cuda:{self.gpuid}')
    else:
      self.device = torch.device(f'cpu')
      print('working on cpu')

    self.model = self.define_net(modelname)
    
    if (not self.isTrain) or (self.isContinue) :
      self.load_networks(loadpath)

    if not self.isTrain:
      self.model.eval() #deactive bn/drop 
    
    if self.isTrain:
      self.criterion = nn.CrossEntropyLoss(reduction='none')
      self.optim = torch.optim.Adam(self.model.parameters(),lr = self.lr)

  def get_device(self):
    return self.device

  #define which network to use
  def define_net(self,modelname):
    net = None
    print('[CREATE] MODEL')
    if(modelname=="UNet"):
      net = UNet(pretrained=True)
    print(modelname)
    # print(net)
    return net.to(self.device)

  #load models from the disk
  def load_networks(self, load_filename):

    load_path = os.path.join(self.save_dir, load_filename)
    checkpoint = torch.load(load_path, map_location=str(self.device))
    self.model.load_state_dict(checkpoint["model_dict"])
    print("[LOADED] MODEL")

  def save_networks(self,model_dir,epoch):

    save_path = os.path.join(self.save_dir, model_dir)
    state = {'epoch': epoch + 1,
          'modelname': self.model_name,
         'model_dict': self.model.state_dict(),
         'optim_dict': self.optim.state_dict(),
                }
    torch.save(state, save_path)

  def set_input(self, X, Y):
    self.X = X.to(self.device)  # [Nbatch, 3, H, W]
    self.Y = Y.type('torch.LongTensor').to(self.device)  # [Nbatch, H, W] with class indices (0, 1)
    
  def update_lr(self):
    self.model.train()
    self.lr = self.lr/2
    self.optim = torch.optim.Adam(self.model.parameters(),lr = self.lr)
    print("update lr to {}".format(self.lr))

  def set_lr(self,rate):
    self.model.train()
    self.lr = rate
    self.optim = torch.optim.Adam(self.model.parameters(),lr = self.lr)
    print("set lr to {}".format(self.lr))

  def set_train(self):
    self.model.train()

  def set_eval(self):
    self.model.eval()

  def forward(self):
    self.pred = self.model(self.X) # [N, Nclass, H, W]
  
  def backward(self):
    self.optim.zero_grad()
    self.loss = self.criterion(self.pred,self.Y).mean() # for one batch
    self.loss.backward()
    self.optim.step()
  
  def get_loss(self):
    return self.loss

  def get_pred(self):
    return self.pred

  def validate(self):
    with torch.no_grad():
      self.forward()
    self.loss = self.criterion(self.pred,self.Y).mean()
    
  def test(self): # only forward 
    with torch.no_grad(): #speed up
      self.forward()