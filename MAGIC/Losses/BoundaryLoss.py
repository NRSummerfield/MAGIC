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

def extract_boundary(mask: torch.Tensor, kernel_size:int =3):
    kernel = torch.ones(tuple([1, 1] + [kernel_size for _ in mask.shape[2:]]), device=mask.device)
    padding = kernel_size // 2

    # looping through the channels to calculate boundaries
    boundaries = []
    for c in range(mask.shape[1]):
        class_mask = mask[:, [c]]
        eroded = F.conv3d(
            input=class_mask.float(), 
            weight=kernel,
            padding=padding,
            ) == kernel.numel()
        boundary = class_mask.bool() & ~eroded
        # if not boundary.sum(): boundary = class_mask
        boundaries.append(boundary.float())
    return torch.cat(boundaries, dim=1)


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

class BoundaryDiceLoss(_Loss):
    def __init__(
        self,
        edge_kernel_size: int = 3,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.edge_kernel_size = edge_kernel_size
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        # Calculate just the boundary of each structure
        input = extract_boundary(input, kernel_size=self.edge_kernel_size)
        target = extract_boundary(target, kernel_size=self.edge_kernel_size)

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


class BoundaryDiceEntropyLoss(_Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss``. In this implementation,
    two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
        self,
        edge_kernel_size:int = 3,
        entropy_type: str = "CrossEntropy",
        dice_type: str = 'Dice',
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
        lambda_boundary_dice: float = 1.0,
        lambda_e: float = 1.0,
        gamma: int = 2,
    ) -> None:
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.boundary_dice = BoundaryDiceLoss(
            edge_kernel_size=edge_kernel_size,
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
            self.entropy_function = cross_entropy(weight=e_weight, reduction=reduction)
        elif entropy_type == "Focal":         
            self.entropy_function = FocalLoss(gamma=gamma, weight=e_weight, reduction=reduction)
        else: raise ValueError(f'{entropy_type=} not found. Known types: ["CrossEntropy", "Focal"].')

        if lambda_boundary_dice < 0.0:
            raise ValueError("lambda_boundary_dice should be no less than 0.0.")
        if lambda_e < 0.0:
            raise ValueError("lambda_e should be no less than 0.0.")
        self.lambda_boundary_dice = lambda_boundary_dice
        self.lambda_e = lambda_e

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        boundary_dice_loss = self.boundary_dice(input, target)
        e_loss = self.entropy_function(input, target)
        total_loss: torch.Tensor = self.lambda_boundary_dice * boundary_dice_loss + self.lambda_e * e_loss

        return total_loss
    
class DiceBoundaryDiceEntropyLoss(_Loss):
    def __init__(
        self,
        edge_kernel_size:int = 3,
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
        lambda_boundary_dice: float = 1.0,
        lambda_e: float = 1.0,
        gamma: int = 2,
    ) -> None:
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.boundary_dice = BoundaryDiceLoss(
            edge_kernel_size=edge_kernel_size,
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
        if lambda_boundary_dice < 0.0:
            raise ValueError("lambda_boundary_dice should be no less than 0.0.")
        if lambda_e < 0.0:
            raise ValueError("lambda_e should be no less than 0.0.")
        self.lambda_boundary_dice = lambda_boundary_dice
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
        boundary_dice_loss = self.boundary_dice(input, target)
        e_loss = self.entropy_function(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_boundary_dice * boundary_dice_loss + self.lambda_e * e_loss

        return total_loss