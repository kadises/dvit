import torch
import torch.nn as nn

import torch
import torch.nn as nn

import numpy as np

class SelfAttention(nn.Module):
  def __init__(self,in_channels,inner_channels,heads = 8,dropout = 0.1):
    super().__init__()
    assert inner_channels % heads == 0, 'wrong channels or heads'
    self.dk = inner_channels // heads
    self.heads = heads
    self.inner_channels = inner_channels
    self.k = nn.Linear(in_features=in_channels,out_features=inner_channels)
    self.q = nn.Linear(in_features=in_channels,out_features=inner_channels)
    self.v = nn.Linear(in_features=in_channels,out_features=inner_channels)

    self.proj = nn.Linear(inner_channels,in_channels)
    self.ln = nn.LayerNorm(in_channels)

    self.do = nn.Dropout(dropout)
    
    
  def forward(self,x):
    # x - [b_s,hw,c] -> [b_s,hw,out_ch] -> [b_s,heads,hw,out_ch]
    bs,hw,_ = x.shape
    
    k = self.k(x).view(bs,self.heads,hw,self.dk)
    q = self.q(x).view(bs,self.heads,hw,self.dk)
    v = self.v(x).view(bs,self.heads,hw,self.dk)

    attention = torch.softmax(q@k.mT/self.dk**(0.5),dim = -1)@v
    # [bs,heads,hw,dk] -> [bs,hw,out_ch]
    attention = attention.view(bs,hw,self.inner_channels)

    attention = self.do(self.proj(attention))

    return attention
  

class CrossAttention(nn.Module):
  def __init__(self,in_channels,cond_channels,inner_channels,heads = 8,dropout = 0.1):
    super().__init__()
    assert inner_channels % heads == 0, 'wrong channels or heads'
    self.dk = inner_channels//heads
    self.heads = heads
    self.inner_channels = inner_channels

    self.k = nn.Linear(cond_channels,inner_channels)
    self.v = nn.Linear(cond_channels,inner_channels)
    self.q = nn.Linear(in_channels,inner_channels)

    self.proj = nn.Linear(inner_channels,in_channels)
    self.do = nn.Dropout(dropout)

  def forward(self,x,cond):
    bs,ps,_ = x.shape
    _,se,_ = cond.shape

    k = self.k(cond).view(bs,self.heads,se,self.dk)
    v = self.v(cond).view(bs,self.heads,se,self.dk)
    q = self.q(x).view(bs,self.heads,ps,self.dk)

    attention = torch.softmax(q@k.mT/self.dk**(0.5),dim = -1)@v
  # [bs,heads,hw,dk] -> [bs,hw,out_ch]
    attention = attention.view(bs,ps,self.inner_channels)

    attention = self.do(self.proj(attention))

    return attention



  

class FeedForwardBlock(nn.Module):
  def __init__(self,in_channels,middle_channels):
    super().__init__()
    self.ff = nn.Sequential(
      nn.Linear(in_features=in_channels,out_features=middle_channels),
      nn.LayerNorm(normalized_shape=middle_channels),
      nn.SiLU(),
      nn.Linear(in_features=middle_channels,out_features=in_channels),
      nn.LayerNorm(normalized_shape=in_channels),
      nn.SiLU(),
    )
    self.ln = nn.LayerNorm(in_channels)

  def forward(self,x):

    x_ = self.ff(x)
    x_ = x_ + self.ln(x)
    return x_
  


class TransofrmerEncoder(nn.Module):
  def __init__(self,in_channels,time_dim,middle_channels,inner_channels,dropout = 0.1,heads = 8):
    super().__init__()
    self.self_att = SelfAttention(in_channels=in_channels,inner_channels=inner_channels,dropout=dropout,heads=heads)
    self.cross_att = CrossAttention(in_channels=in_channels,cond_channels=time_dim,inner_channels=inner_channels,heads = heads,dropout=dropout)
    self.ff = FeedForwardBlock(in_channels=in_channels,middle_channels=middle_channels)
    self.ln1 = nn.LayerNorm(in_channels)
    self.ln2 = nn.LayerNorm(in_channels)
    self.ln3 = nn.LayerNorm(in_channels)
    self.ln4 = nn.LayerNorm(in_channels)

  def forward(self,x,t):
    x = self.self_att(x) + self.ln1(x)

    x = self.cross_att(x,t) + self.ln2(x)

    x = self.ff(x) + self.ln3(x)

    x = self.ln4(x)

    return x
  

class Transformer(nn.Module):
  def __init__(self,in_channels,middle_channels,inner_channels,time_dim,N_blocks=1,dropout = 0.1,heads = 8):
    super().__init__()
    self.blocks = nn.ModuleList([TransofrmerEncoder(in_channels=in_channels,
                                                    middle_channels=middle_channels,
                                                    inner_channels=inner_channels,
                                                    dropout=dropout,
                                                    heads = heads,
                                                    time_dim=time_dim) for _ in range(N_blocks)])
    self.mlp = nn.Sequential(
      nn.Linear(time_dim,middle_channels),
      nn.LayerNorm(middle_channels),
      nn.SiLU(),
      nn.Linear(middle_channels,time_dim),
      nn.LayerNorm(time_dim),
      nn.SiLU()
    )
    self.ln = nn.LayerNorm(in_channels)

  def forward(self,x,time):
    time = self.mlp(time)

    x = self.ln(x)

    for block in self.blocks:
      x = block(x,time)
    
    return x