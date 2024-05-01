import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np

from transformer import VisionTransformer

from diffusion_model import DiffusionModel

from tqdm import tqdm

from utils import get_data,saver,avg,sample_save

def train(args):
  d = DiffusionModel(args)
  
  if args['load_model']:
    d.denoiser.load_state_dict(torch.load(args['weights_path']))

  optim = torch.optim.AdamW(d.denoiser.parameters(),lr = args['lr'],weight_decay=args['weight_decay'])

  criterion = nn.MSELoss()

  print(f'Число параметров: {sum([p.numel() for p in d.denoiser.parameters()])}')

  train = get_data(args['path'],batch_size=args['batch_size'],image_size= args['image_size'])

  for epoch in range(args['epochs']):
    with tqdm(total = len(train)) as t:
      train_loss_avg = []
      d.denoiser.train()
      for imgs,_ in train:
        optim.zero_grad()
        imgs = imgs.to(args['device'])
        times = torch.rand(size=(imgs.shape[0],)).to(args['device'])

        noise = torch.randn_like(imgs)

        signal_rate,noise_rate = d.schedule(times)

        noised_imgs = signal_rate * imgs + noise_rate * noise

        predicted_noise = d.denoiser(noised_imgs,times)

        loss = criterion(noise,predicted_noise)
        train_loss_avg.append(loss)

        avg_loss = avg(train_loss_avg)

        loss.backward()
        optim.step()

        t.set_postfix(loss = loss.item(),avg_loss = avg_loss.item())
        t.update()

      saver(d.denoiser,args['weights_path'])

      #created_imgs = d.sampling(denoiser,args['N_images'],args)

      #sample_save(created_imgs,args['sample_path'],epoch,args['start'])
      

    


args = {'path': '/home/viaznikov/datasets/anime/train_im',
        'lr':1e-4,
        'weight_decay': 1e-3,
        'epochs':100,
        'device':'cuda',
        'batch_size':63,
        'image_size':64,
        'in_channels':3,
        'embedded_channels':128,
        'middle_channels':256,
        'inner_channels':256,
        'patch_size':8,
        'N_blocks':15,
        'time_dim':128,
        'dropout':0.1,
        'heads':8,
        'weights_path':'dvit_0.0.2.pth',
        'load_model':False,
        'N_images':3,
        'sample_path':'/home/viaznikov/nlp/diffusion_transformer/created_images/',
        'start':20,
        'steps':160,
        }

train(args)