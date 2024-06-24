import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

from typing import Optional

class PrepareVITForTrainer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.interpolate_mode = 'nearest'

        if model.model_config.num_labels == 1:
            self.classification_type = 'binary'
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif model.model_config.num_labels > 1:
            self.classification_type = 'multiclass'
            self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, logits, labels):
        if self.classification_type == 'binary':
            return self.loss_fn(logits, labels.float())
        elif self.classification_type == 'multiclass':
            return self.loss_fn(logits, labels)

    def print_trainable_parameters(self):
        trainable_params, all_params = 0, 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f'trainable params: {trainable_params:,} | all params: {all_params:,} | trainable: {100 * trainable_params / all_params}%')

    def forward(self,
                pixel_values: torch.Tensor,
                labels: torch.Tensor):

        logits = self.model(pixel_values)
        return {
            'logits': logits,
            'loss': self.compute_loss(logits.float(), labels)
        }

class PrepareLlamaForTrainer(nn.Module):
    """
    To make the model compatible with HuggingFace's Trainer the model should
    produce a dictionary containing the 'logits' and 'loss'.

    This class implements a custom loss computation and forward pass.
    Also doesn't require any 'label' or 'attention_mask' columns, if any of
    them passed as argument the model will ignore it.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, logits, labels):

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, self.model.n_tokens)
        shift_labels = shift_labels.view(-1).to(torch.long)

        return self.loss_fn(shift_logits, shift_labels)

    def print_trainable_parameters(self):
        trainable_params, all_params = 0, 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f'trainable params: {trainable_params:,} | all params: {all_params:,} | trainable: {100 * trainable_params / all_params}%')

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None):
        logits = self.model(input_ids)
        return {
                'logits': logits,
                'loss': self.compute_loss(logits.float(), input_ids)
            }


class PrepareForCoCaTrainer(nn.Module):
    """
    To make the model compatible with HuggingFace's Trainer the model should
    produce a dictionary containing the 'logits' and 'loss'.

    This class implements a custom loss computation and forward pass.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.ce_loss_fn = F.cross_entropy

    def compute_ca_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, self.model.n_tokens)
        shift_labels = shift_labels.view(-1).to(torch.long)
        #logits = rearrange(logits, 'b n c -> b c n')
        ce = self.ce_loss_fn(shift_logits, shift_labels)
        ca_weight =  self.model.model_config.caption_loss_weight
        ca = ce * ca_weight
        return ca

    def compute_co_loss(self, text_latents, image_latents):
        # calculate contrastive loss
        sim = einsum('i d, j d -> i j', text_latents, image_latents)
        sim = sim * self.model.model_config.temperature.exp()
        contrastive_labels = torch.arange(text_latents.shape[0], device=text_latents.device)

        contrastive_loss = (self.ce_loss_fn(sim, contrastive_labels) + self.ce_loss_fn(sim.t(), contrastive_labels)) * 0.5
        contrastive_loss = contrastive_loss * self.model.model_config.contrastive_loss_weight
        return contrastive_loss

    def compute_loss(self, logits, labels, text_latents, image_latents):
        captioning_loss = self.compute_ca_loss(logits, labels)
        contrastive_loss = self.compute_co_loss(text_latents, image_latents)

        return captioning_loss+contrastive_loss

    def print_trainable_parameters(self):
        trainable_params, all_params = 0, 0
        trainable_params = sum(param.numel() for param in self.image_model.parameters() if param.requires_grad) + \
                sum(param.numel() for param in self.language_model.parameters() if param.requires_grad)

        all_params = sum(param.numel() for param in self.image_model.parameters()) + \
                sum(param.numel() for param in self.language_model.parameters())
        print(f'trainable params: {trainable_params:,} | all params: {all_params:,} | trainable: {100 * trainable_params / all_params}%')

    def forward(self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            text_latents: torch.Tensor,
            image_latents: torch.Tensor
            ):
        return self.compute_loss(logits, labels, text_latents, image_latents)


'''
Apply Embedding to Latents before passing them into the Multimodal Network
'''

class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)
