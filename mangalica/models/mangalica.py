import torch
import torch.nn as nn
import torch.nn.functional as F

class MangaliCaConfig():
    def __init__(self,
             text_dim: int = 1024,
             image_dim: int = 768,
             num_img_query: int = 256,
             caption_loss_weight: int = 1.0,
             contrastive_loss_weight: int = 1.0):
                 
        self.text_dim = text_dim
        self.img_dim = img_dim
        self.num_img_query = num_img_query

class MangaliCa(nn.Module):
    def __init__(self,
                 img_encoder = None,
                 unimodal_decoder = None,
                 multimodal_decoder = None,
                 config: MangaliCaConfig):
        self.model_config = config

        self.img_encoder = img_encoder
        self.unimodal_decoder = unimodal_decoder
        self.multimodal_decoder = multimodal_decoder
