import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn

import numpy as np

from modules import Transformer


class VisionTransformer(nn.Module):
  def __init__(self,in_channels,middle_channels,embedded_channels,inner_channels,patch_size,N_blocks=1,time_dim=128,dropout=0.1,heads = 8):
    super().__init__()

    self.time_dim = time_dim
    
    self.patcher = nn.Sequential(
      nn.Conv2d(in_channels=in_channels,
                out_channels=embedded_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0),
      nn.BatchNorm2d(embedded_channels),
      nn.Flatten(start_dim=2)
    )
    #x - [bs,e,ps^2]
    self.pos = nn.Embedding(patch_size**2,embedding_dim=embedded_channels)

    self.transformer = Transformer(
      in_channels=embedded_channels,
      middle_channels=middle_channels,
      inner_channels=inner_channels,
      N_blocks=N_blocks,
      dropout=dropout,
      heads=heads,
      time_dim=time_dim
    )

    self.mlp = nn.Sequential(
      nn.Linear(embedded_channels,middle_channels),
      nn.LayerNorm(middle_channels),
      nn.SiLU(),
      nn.Linear(middle_channels,embedded_channels),
      nn.LayerNorm(embedded_channels),
      nn.SiLU()
    )
    

    self.out = nn.ConvTranspose2d(in_channels=embedded_channels,
                                  out_channels=in_channels,
                                  kernel_size=patch_size,
                                  stride=patch_size,
                                  padding=0)
    
  def embedding(self,t, channels):
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, channels, 2).float() / channels)
    ).to(t.device)
    pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc
    

  def forward(self,x,time):
    x = self.patcher(x)
    bs,e,ps = x.shape
    #print(f'x: {x.shape}')
    x = x.view(bs,ps,e)

    pos = self.pos(torch.arange(0,ps,dtype=torch.int64,device=x.device))

    x += pos
    time = time.unsqueeze(1)

    time = self.embedding(time,self.time_dim).unsqueeze(1)
    time = time.repeat(1,64,1)

    x = self.transformer(x,time)
    
    x = self.mlp(x)

    sqrt_ps = int(math.sqrt(ps))

    x = x.view(bs,e,sqrt_ps,sqrt_ps)

    x = self.out(x)

    return x