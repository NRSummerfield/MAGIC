from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm, Conv
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks import UnetBasicBlock
from monai.utils import InterpolateMode, UpsampleMode, deprecated_arg, ensure_tuple_rep, look_up_option

def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )

def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


def pass_forward(x): return x

class SplitUnetResBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: Sequence[int],
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        out_channels = out_channels if not isinstance(out_channels, int) else [out_channels]
        self.num_repeats = len(out_channels)
        conv1 = [get_conv_layer(spatial_dims, in_channels, out_channel, kernel_size=kernel_size, stride=stride, dropout=dropout, conv_only=True) for out_channel in out_channels]
        self.conv1 = nn.ModuleList(conv1)
    
        conv2 = [get_conv_layer(spatial_dims, out_channel, out_channel, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True) for out_channel in out_channels]
        self.conv2 = nn.ModuleList(conv2)

        self.lrelu = get_act_layer(name=act_name)
        
        norm1 = [get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channel) for out_channel in out_channels]
        norm2 = [get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channel) for out_channel in out_channels]
        self.norm1 = nn.ModuleList(norm1)
        self.norm2 = nn.ModuleList(norm2)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            conv3 = [get_conv_layer(spatial_dims, in_channels, out_channel, kernel_size=1, stride=stride, dropout=dropout, conv_only=True) for out_channel in out_channels]
            self.conv3 = nn.ModuleList(conv3)
            norm3 = [get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channel) for out_channel in out_channels]
            self.norm3 = nn.ModuleList(norm3)

    def forward(self, inp):
        # if input is a tensor, repeat that input for all branches
        # if it is a list or tuple, go through the sequence of for each tensor
        repeat = isinstance(inp, torch.Tensor)

        all_out = [] # recording all outputs
        for i in range(self.num_repeats):
            # Checking if you need to repeat the input or go to next tensor
            if repeat: _inp = inp
            else: _inp = inp[i]

            # Go through all layers
            residual = _inp
            out = self.conv1[i](_inp)
            out = self.norm1[i](out)
            out = self.lrelu(out)
            out = self.conv2[i](out)
            out = self.norm2[i](out)

            # Go through possible 3rd layer
            if hasattr(self, "conv3"):
                residual = self.conv3[i](residual)
            if hasattr(self, "norm3"):
                residual = self.norm3[i](residual)

            out += residual
            out = self.lrelu(out)
            all_out.append(out)
        return all_out
    
class SplitUnetBasicBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: Sequence[int],
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.num_repeats = len(out_channels)
        conv1 = [get_conv_layer(spatial_dims, in_channels, out_channel, kernel_size=kernel_size, stride=stride, dropout=dropout, conv_only=True) for out_channel in out_channels]
        self.conv1 = nn.ModuleList(conv1)
    
        conv2 = [get_conv_layer(spatial_dims, out_channel, out_channel, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True) for out_channel in out_channels]
        self.conv2 = nn.ModuleList(conv2)

        self.lrelu = get_act_layer(name=act_name)

        norm1 = [get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channel) for out_channel in out_channels]
        norm2 = [get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channel) for out_channel in out_channels]
        self.norm1 = nn.ModuleList(norm1)
        self.norm2 = nn.ModuleList(norm2)

    def forward(self, inp):
        # if input is a tensor, repeat that input for all branches
        # if it is a list or tuple, go through the sequence of for each tensor
        repeat = isinstance(inp, torch.Tensor)

        all_out = [] # recording all outputs
        for i in range(self.num_repeats):
            # Checking if you need to repeat the input or go to next tensor
            if repeat: _inp = inp
            else: _inp = inp[i]

            # Go through all layers
            out = self.conv1[i](_inp)
            out = self.norm1[i](out)
            out = self.lrelu(out)
            out = self.conv2[i](out)
            out = self.norm2[i](out)
            out = self.lrelu(out)
            all_out.append(out)
        return all_out
    
class SplitUnetUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: Sequence[int],
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        self.num_repeats = len(out_channels)
        upsample_stride = upsample_kernel_size

        transp_conv = [get_conv_layer(spatial_dims, in_channels, out_channel, kernel_size=upsample_kernel_size, stride=upsample_stride, dropout=dropout, bias=trans_bias, conv_only=True, is_transposed=True) for out_channel in out_channels]
        self.transp_conv = nn.ModuleList(transp_conv)

        conv_block = [UnetBasicBlock(spatial_dims, out_channel + out_channel, out_channel, kernel_size=kernel_size, stride=1, dropout=dropout, norm_name=norm_name, act_name=act_name) for out_channel in out_channels]
        self.conv_block = nn.ModuleList(conv_block) 

    def forward(self, inp, skip):
        # if input is a tensor, repeat that input for all branches
        # if it is a list or tuple, go through the sequence of for each tensor
        repeat = isinstance(inp, torch.Tensor)

        all_out = [] # recording all outputs
        for i in range(self.num_repeats):
            # Checking if you need to repeat the input or go to next tensor
            if repeat: _inp = inp
            else: _inp = inp[i]

            # number of channels for skip should equals to out_channels
            out = self.transp_conv[i](_inp)
            out = torch.cat((out, skip), dim=1)
            out = self.conv_block[i](out)
            all_out.append(out)
        return all_out

class SplitUnetOutBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: list[int], dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        conv = [get_conv_layer(spatial_dims, in_channels, out_channel, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True) for out_channel in out_channels]
        self.conv = nn.ModuleList(conv)
        self.num_repeats = len(out_channels)

    def forward(self, inp):
        repeat = isinstance(inp, torch.Tensor)
        all_out = []

        for i in range(self.num_repeats):
            if repeat: _inp = inp
            else: _inp = inp[i]

            out = self.conv[i](_inp)
            all_out.append(out)

        return all_out
    

__all__ = ["Upsample", "UpSample", "SubpixelUpsample", "Subpixelupsample", "SubpixelUpSample"]


class SplitUpSample(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: Sequence[int],
            scale_factor: Union[Sequence[float], float] = 2,
            bias: bool = True,
            ):
        super().__init__()
        self.num_repeats = len(out_channels)
        scale_factor_ = ensure_tuple_rep(scale_factor, spatial_dims)

        deconvs = [Conv[Conv.CONVTRANS, spatial_dims](in_channels=out_channel, out_channels=out_channel, kernel_size=scale_factor_, stride=scale_factor_, bias=bias) for out_channel in out_channels]
        self.deconvs = nn.ModuleList(deconvs)
    
    def forward(self, inp):
        repeat = isinstance(inp, torch.Tensor)
        all_out = []
        for i in range(self.num_repeats):
            if repeat: _inp = inp
            else: _inp = inp[i]

            out = self.deconvs[i](_inp)
            all_out.append(out)

        return(all_out)