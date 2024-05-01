import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformer import VisionTransformer

class DiffusionModel:
  def __init__(self,args):
    self.denoiser = VisionTransformer(in_channels=args['in_channels'],
                                      embedded_channels=args['embedded_channels'],
                                      middle_channels=args['middle_channels'],
                                      inner_channels=args['inner_channels'],
                                      patch_size=args['patch_size'],
                                      N_blocks=args['N_blocks'],
                                      time_dim=args['time_dim'],
                                      dropout=args['dropout'],
                                      heads=args['heads'],
                                      ).to(args['device'])
    
    self.device = args['device']
    self.in_channels = args['in_channels']
    self.batch_size = args['batch_size']
    self.image_size = args['image_size'] 
    self.time_dim = args['time_dim']

  def schedule(self,time):
    angle = (torch.pi/2*time).to(self.device)

    signal_rate = torch.cos(angle)[:,None,None,None]
    noise_rate = torch.sin(angle)[:,None,None,None]

    return signal_rate,noise_rate
  
  
  
  def denoise(self,time,noised_image,signal_rate,noise_rate):
    noise = self.denoiser(noised_image,time)
    image = (noised_image - noise_rate*noise)/signal_rate

    return image,noise
  
  def sample(self,N_images,steps):
    step = torch.tensor(1/steps)
    times = torch.arange(start=0,end=1,step=step)

    imgs = torch.randn(size = (N_images,self.in_channels,self.image_size,self.image_size)).to(self.device)

    self.denoiser.eval()
    one = torch.ones(size = (N_images,))
    self.denoiser.zero_grad()
    
    with tqdm(total = len(times) - 1) as tq:
      for time in reversed(times[1:]):
        t = (time*one).to(self.device)

        signal_rate,noise_rate = self.schedule(t)

        pred_imgs,pred_noise = self.denoise(t,imgs,signal_rate,noise_rate)

        del t
        time_ = time - step

        t_ = (time_*one).to(self.device)

        signal_rate_,noise_rate_ = self.schedule(t_)

        del t_,signal_rate,noise_rate
        imgs_ = signal_rate_*pred_imgs + noise_rate_*pred_noise
        
        imgs = imgs_

        del imgs_,signal_rate_,noise_rate_
        tq.update()

    return pred_imgs



