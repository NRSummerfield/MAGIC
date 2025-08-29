import warnings
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.losses.dice import DiceLoss, GeneralizedDiceLoss
from monai.losses.focal_loss import FocalLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, look_up_option

from .cldice_loss import SoftclDiceLoss
from .cbdice_loss import SoftcbDiceLoss


class cross_entropy(nn.CrossEntropyLoss):
    def __init__(self, weight, reduction):
        super().__init__(weight=weight, reduction=reduction)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()
        return super().forward(input, target)

class DiceCenterLineDiceEntropyLoss(_Loss):
    def __init__(
        self,
        entropy_type: str = "CrossEntropy",
        dice_type:str = 'Dice',
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        e_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_CB_dice: float = 1.0,
        lambda_e: float = 1.0,
        gamma: int = 2,
        center_line_iter=10,
        center_line_smooth=1,
        center_line_default_topology=True
    ) -> None:
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.center_line_loss = SoftcbDiceLoss(
            iter_=center_line_iter,
            smooth=center_line_smooth,
            default_topology=center_line_default_topology
        )

        if dice_type == 'General':
            self.dice = GeneralizedDiceLoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                sigmoid=sigmoid,
                softmax=softmax,
                other_act=other_act,
                # w_type=..., # Default = square -> Good for class imbalance
                reduction=reduction,
                smooth_nr=smooth_nr,
                smooth_dr=smooth_dr,
                batch=batch
            )
        elif dice_type == 'Dice':
            self.dice = DiceLoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                sigmoid=sigmoid,
                softmax=softmax,
                other_act=other_act,
                squared_pred=squared_pred,
                jaccard=jaccard,
                reduction=reduction,
                smooth_nr=smooth_nr,
                smooth_dr=smooth_dr,
                batch=batch,
            )
        if entropy_type == "CrossEntropy":
            self.cross_entropy = nn.CrossEntropyLoss(weight=e_weight, reduction=reduction)
            self.entropy_function = self.ce
        elif entropy_type == "Focal":         
            self.entropy_function = FocalLoss(gamma=gamma, weight=e_weight, reduction=reduction, to_onehot_y=to_onehot_y)
        else: raise ValueError(f'{entropy_type=} not found. Known types: ["CrossEntropy", "Focal"].')

        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_CB_dice < 0.0:
            raise ValueError("lambda_CB_dice should be no less than 0.0.")
        if lambda_e < 0.0:
            raise ValueError("lambda_e should be no less than 0.0.")
        self.lambda_CB_dice = lambda_CB_dice
        self.lambda_e = lambda_e
        self.lambda_dice = lambda_dice
    
    def ce(self, input: torch.Tensor, target: torch.Tensor):
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()
        return self.cross_entropy(input, target)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        dice_loss = self.dice(input, target)
        center_line_dice_loss = self.center_line_loss(input, target)
        e_loss = self.entropy_function(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_CB_dice * center_line_dice_loss + self.lambda_e * e_loss

        return total_loss