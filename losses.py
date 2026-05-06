# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
Adaptive distillation strategy inspired by DeiT and DearKD
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.

    Supports adaptive distillation where alpha changes during training based on:
    - DeiT: soft and hard distillation with distillation token
    - DearKD: early-stage stronger distillation for data efficiency
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard', 'soft_hard', 'adaptive_soft_hard']
        self.distillation_type = distillation_type
        self.alpha = alpha  # This will be the current alpha value
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        # Compute distillation loss based on type
        if self.distillation_type in ['soft', 'soft_hard', 'adaptive_soft_hard']:
            T = self.tau
            # Soft distillation using KL divergence (from DeiT)
            soft_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()

            if self.distillation_type == 'soft':
                distillation_loss = soft_loss
            else:
                # Soft-hard joint distillation
                hard_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
                # Combine soft and hard losses (equal weight)
                distillation_loss = 0.5 * soft_loss + 0.5 * hard_loss

        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        # Combine base loss and distillation loss with current alpha
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

    def set_alpha(self, new_alpha):
        """
        Update the distillation weight alpha during training.
        Used for adaptive distillation strategy.

        Args:
            new_alpha: new distillation weight (0-1)
        """
        self.alpha = new_alpha
