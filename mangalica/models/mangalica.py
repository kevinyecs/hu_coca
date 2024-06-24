import torch
import torch.nn as nn
import torch.nn.functional as F

class MangaliCaConfig():
    def __init__(self,
             text_dim: int = 1024,
             image_dim: int = 768,
             num_img_query: int = 256,
             caption_loss_weight: int = 1.0,
             contrastive_loss_weight: int = 1.0,
             temperature: float = 0.07
             ):

        self.text_dim = text_dim
        self.img_dim = image_dim
        self.num_img_query = num_img_query
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.temperature = temperature

class MangaliCa(nn.Module):
    def __init__(self,
                 config: MangaliCaConfig,
                 img_encoder = None,
                 unimodal_decoder = None,
                 multimodal_decoder = None,
                 ):
        self.model_config = config

        self.img_encoder = img_encoder
        self.unimodal_decoder = unimodal_decoder
        self.multimodal_decoder = multimodal_decoder
