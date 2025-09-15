import numpy as np, torch
from typing import Optional   

class GroupSeperate:
    """Seperates an input tensor into seperate tensors of specific channel-based groups"""
    def __init__(self, idxs_groups: list[list[int]]): 
        self.idxs_groups = idxs_groups

    def __call__(self, arr: torch.Tensor) -> tuple[torch.Tensor]:
        return [arr[:, igrp] for igrp in self.idxs_groups]

class lr_lambda:
    """Simple class calculate a new learning rate following a polynomial decay"""
    def __init__(self, LRS_converge_epoch, gamma):
        self.lrs_ce = LRS_converge_epoch
        self.g = gamma
    def __call__(self, epoch):
        return (1 - epoch / self.lrs_ce) ** self.g    
        

def format_training_output_nnUNetwSD(tree_output: tuple[dict, list[dict]]) -> dict:
    """
    Formats an output of a multi-branch tree for use in a loss function

    Expected input format (from MAGIC's training output):
        dict - output of the decoder branch
            'out': torch.Tensor
            'teacher': torch.Tensor
            'students': list[torch.Tensor]
        tuple - output of each encoder(s)
            dict - output of a encoder branch
                'out': torch.Tensor
                'teacher': torch.Tensor
                'students': list[torch.Tensor]
    
    Output format:
        dict: {
            'out': torch.Tensor
            'decoder': {
                'teacher': torch.Tensor
                'students': list[torch.Tensor]
                }
            'encoder': {
                'teacher': torch.Tensor
                'students': list[torch.Tensor]
                }
    """
    output = {}
    decoder_output, encoder_output = tree_output
    output['out'] = decoder_output['out']
    output['encoder'] = {'students': encoder_output['students'], 'teacher': encoder_output['teacher']}
    output['decoder'] = {'students': decoder_output['students'], 'teacher': decoder_output['teacher']}
    return output


class split_groups_transform:
    """Dictionary based transform that will split a channeled label into different grouped labels"""
    def __init__(self, target_key: str, group_idxs: list[list[int]], out_names: list[str], stacking_order: Optional[list[list[int]]] = None):
        self.tk = target_key
        self.group_idxs = group_idxs
        self.out_names = out_names
        self.stacking_order = stacking_order if stacking_order is not None else [[i for i in range(len(g))] for g in group_idxs]

    def __call__(self, d: dict):
        d = dict(d)
        src_arr = d[self.tk]
        for gidx, gname, gorder in zip(self.group_idxs, self.out_names, self.stacking_order):
            base = np.zeros(src_arr.shape[1:]) if isinstance(src_arr, np.ndarray) else torch.zeros(src_arr.shape[1:], dtype=src_arr.dtype)
            glabel = src_arr[gidx]
            for i in gorder:
                base[glabel[i] == 1] = i + 1
            d[gname] = base[None]
        return d
    
class channel_to_stacked_binary_transform:
    """Dictionary based transform that will stack all the labels and binarize them for a uniform basis"""
    def __init__(self, target_key: str, out_key: str):
        self.tk = target_key
        self.ok = out_key
    
    def __call__(self, d: dict):
        d = dict(d)
        base = d[self.tk]
        d[self.ok] = base.sum(0)[None] > 1
        return d
    