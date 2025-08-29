import torch, numpy as np, nibabel as nib, scipy.ndimage as ndimg, os, glob, skimage as skimg, gudhi
from typing import Optional, Any, Callable
from Stuff.BBox import BBox_functions as bbfun
from skimage.morphology import skeletonize

_e = 1e-6

class ChannelMetric:
    def __init__(self, ChannelFunction: Callable[[torch.Tensor | np.ndarray], float], independent: bool = True, function_kwargs: dict = {}):
        self.f = ChannelFunction
        self.indep = independent
        self.history = []
        self.f_kwargs = function_kwargs

    def calculate(self, prediction: torch.Tensor | np.ndarray, reference: torch.Tensor | np.ndarray):
        if self.indep: out = [self.f(r, **self.f_kwargs) - self.f(p, **self.f_kwargs) for p, r in zip(prediction, reference)]
        else: out = [self.f(p, r, **self.f_kwargs) for p, r in zip(prediction, reference)]
        self.history.append(out)
        return out
        
    def reset(self):
        self.history = []

    def aggregate(self):
        return np.array(self.history)

    def __call__(self, prediction, reference):
        self.calculate(prediction=prediction, reference=reference)

def volume_ratio(prediction: np.ndarray | torch.Tensor, reference: np.ndarray | torch.Tensor, epsilon: Optional[float] = None):
    _e = epsilon if epsilon is not None else 0
    if isinstance(prediction, torch.Tensor): prediction = prediction.cpu().numpy()
    if isinstance(reference, torch.Tensor): reference = reference.cpu().numpy()

    return prediction.sum() / (reference.sum() + _e)

def compute_betti(mask: np.ndarray | torch.Tensor):
    if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
    if not mask.sum(): return np.asarray([0 for _ in range(mask.ndim + 1)])
    mask = bbfun.crop_to_bbox(mask, bbox=bbfun.bounding_box(mask))
    mask = 1 - mask

    cubical_complex = gudhi.CubicalComplex(top_dimensional_cells = mask)

    betti_nums = [0 for _ in range(mask.ndim)]
    for dim, (birth, death) in cubical_complex.persistence():
        if birth <= 0 and (death == float('inf') or death > 0): betti_nums[dim] += 1

    betti_nums.append(betti_nums[0] - betti_nums[1] + betti_nums[2])
    return np.asarray(betti_nums)

def center_line_score(mask, skel):
    return np.sum(mask * skel) / (np.sum(skel) + _e)

def center_line_Dice(prediction, reference):
    pred_over_label_skeleton = center_line_score(prediction, skeletonize(reference))
    label_over_pred_skeleton = center_line_score(reference, skeletonize(prediction))

    return 2 * pred_over_label_skeleton * label_over_pred_skeleton / (pred_over_label_skeleton + label_over_pred_skeleton + _e)

def SimpleDice(prediction, reference):
    if isinstance(prediction, torch.Tensor): prediction = prediction.cpu().numpy()
    if isinstance(reference, torch.Tensor): reference = reference.cpu().numpy()

    return 2 * (prediction * reference).sum() / (prediction.sum() + reference.sum() + _e)


class BettiMetric(ChannelMetric):
    def __init__(self):
        super().__init__(ChannelFunction=compute_betti, independent=True)
    
class clDiceMetric(ChannelMetric):
    def __init__(self):
        super().__init__(ChannelFunction=center_line_Dice, independent=False)

class VolumeMetric(ChannelMetric):
    def __init__(self, epsilon: Optional[float] = None):
        super().__init__(ChannelFunction=volume_ratio, independent=False, function_kwargs=dict(epsilon=epsilon))
    
class DiceMetric(ChannelMetric):
    def __init__(self):
        super().__init__(ChannelFunction=SimpleDice, independent=False)
        


# Reimplitation of Monai functions
