import torch
import numpy as np
from torchvision import transforms,datasets
from PIL import Image

def get_data(path,batch_size,image_size):
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((image_size,image_size)),
    transforms.Normalize((0.5),(0.5))
  ])

  train_data = datasets.ImageFolder(path,transform=transform)

  train = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

  return train


def avg(li,device = 'cuda'):
  return torch.tensor(li).mean(dim=-1).to(device)

def saver(model,weights_path):
  torch.save(model.state_dict(),weights_path)


def sample_save(imgs,path,iter,start):
  bs,c,h,w = imgs.shape
  imgs = imgs.view(c,h,w*bs)
  images = transforms.ToPILImage()(imgs)
  iter += start
  images.save(path + f'/{iter}_epoch.png')

  del imgs,images







