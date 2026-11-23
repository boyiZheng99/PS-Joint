


"""
DECO ConvNet classes.
"""

import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import math
import numpy as np
from timm.models.layers import DropPath


class DECO_ConvNet(nn.Module):
    '''DECO ConvNet class, including encoder and decoder'''

    def __init__(self, 
                 num_queries=100, 
                 d_model=512,  
                 num_decoder_layers=6,
                 normalize_before=False, 
                 return_intermediate_dec=False, 
                 qH=10,
                 feature_map_size=192,
                 patch_size=8,
                 in_channels=256,
                 ):
        super().__init__()

        # object query shape
        self.qH = qH
        self.qW = int(float(num_queries)/float(self.qH))
        print('query shape {}x{}'.format(self.qH, self.qW))  

        # # encoder
        # self.encoder = DecoEncoder(enc_dims=enc_dims, enc_depth=enc_depth)  
        # 
        self.patch_embed = PatchEmbed(img_size=feature_map_size, patch_size=patch_size, in_c=in_channels, embed_dim=d_model)   

        # decoder
        decoder_layer = DecoDecoderLayer(d_model, normalize_before, self.qH, self.qW)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = DecoDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, qH=self.qH, qW=self.qW)

        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_embed = nn.Linear(d_model, d_model)

        self.mask_embed = MLP(d_model, d_model, in_channels, 3) 

        # other initialization
        # self.tgt = nn.Embedding(num_queries, d_model)
        self._reset_parameters()
        self.d_model = d_model
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, text_emb=None):

        mask_features =  src

        bs, c, h, w = src.shape

        src = self.patch_embed(src)

        if text_emb is None:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            hs = self.decoder(query_embed, src, bs=bs, d_model=self.d_model)  #随机生成一组query
        else:
            hs = self.decoder(text_emb, src, bs=bs, d_model=self.d_model)  #随机生成一组query
        

        cls_suff = self.class_embed(hs[-1])

        mask_embed = self.mask_embed(hs[-1])   

        cls_seg_intermask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # tgt=self.tgt.weight.unsqueeze(1).repeat(1, bs, 1)
        # memory = self.encoder(src)
       
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # query_embed = query_embed.permute(1, 2, 0).view(bs, self.d_model,self.qH,self.qW)

        # hs = self.decoder(tgt, memory, bs=bs, d_model=self.d_model, query_pos=query_embed) #1*（h*w)*b*c
        
        return cls_suff, cls_seg_intermask #hs:1*b*(h*w)*c



class DecoDecoder(nn.Module):
    '''Define Deco Decoder'''
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, qH=10, qW=10):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.qH = qH
        self.qW = qW

    def forward(self, tgt, memory, bs, d_model):
        
        output = torch.zeros_like(tgt)
        query_pos = tgt.permute(1, 2, 0).reshape(bs, d_model,self.qH,self.qW)

        intermediate = []

        for layer in self.layers:
            output=output.permute(1, 2, 0).reshape(bs, d_model,self.qH,self.qW)
            output = layer(output, memory, query_pos=query_pos)
            output=output.flatten(2).permute(2, 0, 1)  #（h*w)*b*c
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output.unsqueeze(0).permute(0,2,1,3)  #1*（h*w)*b*c

class DecoDecoderLayer(nn.Module):
    '''Define a layer for Deco Decoder'''
    def __init__(self,d_model, normalize_before=False, qH=10, qW=10,
                 drop_path=0.,layer_scale_init_value=1e-6):
        super().__init__()
        self.normalize_before = normalize_before
        self.qH = qH
        self.qW = qW

        # The SIM module   
        self.dwconv1 = nn.Conv2d(d_model, d_model, kernel_size=9, padding=4, groups=d_model) 
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.pwconv1_1 = nn.Linear(d_model, 4 * d_model) 
        self.act1 = nn.GELU()
        self.pwconv1_2 = nn.Linear(4 * d_model, d_model)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((d_model)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # The CIM module
        self.dwconv2 = nn.Conv2d(d_model, d_model, kernel_size=9, padding=4, groups=d_model) 
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.pwconv2_1 = nn.Linear(d_model, 4 * d_model) 
        self.act2 = nn.GELU()
        self.pwconv2_2 = nn.Linear(4 * d_model, d_model)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((d_model)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, tgt, memory, query_pos: Optional[Tensor] = None):
        # SIM
        b, d, h, w = memory.shape
        tgt2 = tgt + query_pos
        tgt2 = self.dwconv1(tgt2)
        tgt2 = tgt2.permute(0, 2, 3, 1) # (b,d,10,10)->(b,10,10,d)
        tgt2 = self.norm1(tgt2)
        tgt2 = self.pwconv1_1(tgt2)
        tgt2 = self.act1(tgt2)
        tgt2 = self.pwconv1_2(tgt2)
        if self.gamma1 is not None:
            tgt2 = self.gamma1 * tgt2
        tgt2 = tgt2.permute(0,3,1,2) # (b,10,10,d)->(b,d,10,10)
        tgt = tgt + self.drop_path1(tgt2)

        # CIM
        tgt = F.interpolate(tgt, size=[h,w]) 
        tgt2 = tgt + memory 
        tgt2 = self.dwconv2(tgt2)
        tgt2 = tgt2+tgt 
        tgt2 = tgt2.permute(0, 2, 3, 1) # (b,d,h,w)->(b,h,w,d)
        tgt2=self.norm2(tgt2)
        
        # FFN
        tgt = tgt2
        tgt2 = self.pwconv2_1(tgt2)
        tgt2 = self.act2(tgt2)
        tgt2 = self.pwconv2_2(tgt2)
        if self.gamma2 is not None:
            tgt2 = self.gamma2 * tgt2
        tgt2 = tgt2.permute(0,3,1,2) # (b,h,w,d)->(b,d,h,w)
        tgt = tgt.permute(0,3,1,2) # (b,h,w,d)->(b,d,h,w)
        tgt = tgt + self.drop_path1(tgt2)

        # pooling
        m = nn.AdaptiveMaxPool2d((self.qH, self.qW))
        tgt = m(tgt)

        return tgt

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.norm(self.proj(x))
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
