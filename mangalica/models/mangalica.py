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
        self.img_dim = image_dim
        self.num_img_query = num_img_query
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

class MangaliCa(nn.Module):
    """
    MangaliCa (Mangalica CoCa)

    Detailed Description
    """
    
    def __init__(self,
                 config: MangaliCaConfig,
                 img_encoder = None,
                 unimodal_decoder = None,
                 multimodal_decoder = None):
        self.model_config = config

        self.img_encoder = img_encoder
        self.unimodal_decoder = unimodal_decoder
        self.multimodal_decoder = multimodal_decoder

        ## Needs an import and initialization
        self.attn_pooling = XAttn(...)

        ## (captioning embeddings, contrastive embeddings)
        self.img_query = nn.Parameter(torch.randn(num_img_query + 1, dim))

    def forward(self,
                pixel_values: torch.Tensor,
                input_ids: torch.Tensor):
        device = pixel_values.device

        img_feature = self.img_encoder(pixel_values)
        cap_feature, con_feature = self.attn_pooling(img_feature, self.img_query).split([num_img_query, 1])

        
        
