import torch
import torch.nn as nn


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