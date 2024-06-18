import torch
import torch.nn as nn
from typing import Optional


class PrepareForImageTrainer(nn.Module):
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
    



class PrepareForNLPTrainer(nn.Module):
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