# pyright: reportPrivateImportUsage=false
from typing import List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from monai.utils import UpsampleMode, InterpolateMode

# Relative import for final training model
from .deepUp import DeepUp

# Absolute import for testing this script
# from deepUp import DeepUp

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock, UnetUpBlock

def get_module_list(
        conv_block: Type[nn.Module],
        spatial_dims: int,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
        kernel_sizes: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str],
        dropout: Optional[Union[Tuple, str, float]],
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
        trans_bias: bool = False,
        ):
    """
    Returns a module list of cascading layers that for use in a CNN.

    If upsample_kernel_size is not None, then this is an upsampling layer (Decoder)
    If upsample_kernel_size is None, then this is a downsampling layer (Encoder)

    returns: type[nn.ModuleList(layers)]
    """
    # initialize a list to hold the layers
    layers = []
    # if upsample is None, make an interable sequence of Nones
    upsample_kernel_size = [None for _ in in_channels] if upsample_kernel_size is None else upsample_kernel_size
    # Go through the paramters
    for in_c, out_c, kernel, stride, up_kernel in zip(in_channels, out_channels, kernel_sizes, strides, upsample_kernel_size):
        # configure a baseline parameter dictionary for this layer
        params = dict(spatial_dims = spatial_dims, in_channels = in_c, out_channels = out_c, kernel_size = kernel, stride = stride, norm_name = norm_name, act_name = act_name, dropout = dropout)
        # if upsample_kerenel_size is not None, then it is an upsample layer -> add upsample_kernel_size and trans_bias arguments
        if up_kernel is not None: params = params | dict(upsample_kernel_size = up_kernel, trans_bias = trans_bias)
        layers.append(conv_block(**params))
    # convert to a torch module list
    return nn.ModuleList(layers)


class DynEncoder(nn.Module):
    def __init__(self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            filters: Sequence[int],
            kernel_size: Sequence[Union[Sequence[int], int]],
            strides: Sequence[Union[Sequence[int], int]],
            norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
        ):
        super().__init__()

        # Arguments that describe the convolutions
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout 

        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsamples()
        
    def get_input_block(self):
        return UnetResBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.filters[0],
            kernel_size=self.kernel_size[0],
            stride=self.strides[0],
            norm_name=self.norm_name,
            act_name=self.act_name,
            dropout=self.dropout,
        )
    
    def get_downsamples(self):
        # Note:
        # Input : downsamples : Bottleneck
        # [0]   :    [1:-1]   :    [-1]
        return get_module_list(
            conv_block = UnetResBlock,
            spatial_dims = self.spatial_dims,
            in_channels = self.filters[:-2],
            out_channels = self.filters[1:-1],
            kernel_sizes = self.kernel_size[1:-1],
            strides = self.strides[1:-1],
            norm_name = self.norm_name,
            act_name = self.act_name,
            dropout = self.dropout,
            upsample_kernel_size = None,
            trans_bias = False,
        )
    
    def forward(self, x: torch.Tensor) -> dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]]:
        out = self.input_block(x)
        bridges = [out]
        for module in self.downsamples:
            out = module(out)
            bridges.append(out)
        return {'to_bottle': out, 'bridges': bridges}


class DynDecoder(nn.Module):
    def __init__(self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            filters: Sequence[int],
            kernel_size: Sequence[Union[Sequence[int], int]],
            strides: Sequence[Union[Sequence[int], int]],
            norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
        ):
        super().__init__()

        # Arguments that describe the convolutions
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout 

        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block()
        
    def get_bottleneck(self):
        return UnetResBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.filters[-2],
            out_channels=self.filters[-1],
            kernel_size=self.kernel_size[-1],
            stride=self.strides[-1],
            norm_name=self.norm_name,
            act_name=self.act_name,
            dropout=self.dropout,
        )
    
    def get_output_block(self):
        return UnetOutBlock(
            spatial_dims=self.spatial_dims, 
            in_channels=self.filters[0], 
            out_channels=self.out_channels, 
            dropout=self.dropout)
        
    def get_upsamples(self):
        # Note:
        # Bottleneck : Upsamples : Output
        #  [-1 ...]  :    [1:]   :   [0]
        return get_module_list(
            conv_block = UnetUpBlock,
            spatial_dims = self.spatial_dims,
            in_channels = self.filters[1:][::-1],
            out_channels = self.filters[:-1][::-1],
            kernel_sizes = self.kernel_size[1:][::-1],
            strides = self.strides[1:][::-1],
            norm_name = self.norm_name,
            act_name = self.act_name,
            dropout = self.dropout,
            upsample_kernel_size = self.strides[1:][::-1],
            trans_bias = False,
        )
    
    def forward(self, x: Union[Sequence[dict], dict]) -> dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]]:
        # if multiple inputs are given, average them and continue
        if isinstance(x, (tuple, list)):
            to_bottle: torch.Tensor = torch.stack([_dict['to_bottle'] for _dict in x]).mean(0)
            bridges: list[torch.Tensor] = [torch.stack(_set).mean(0) for _set in zip(*[_dict['bridges'] for _dict in x])]
        elif isinstance(x, dict):
            to_bottle: torch.Tensor = x['to_bottle']
            bridges: list[torch.Tensor] = x['bridges']

        # reverse the bridges because it builds it back UP
        bridges.reverse()
    
        out = self.bottleneck(to_bottle)
        for module, bridge in zip(self.upsamples, bridges):
            out = module(out, bridge)
    
        return self.output_block(out)


if __name__ == '__main__':
    input1 = torch.rand([1, 1, 96, 96, 96])
    input2 = torch.rand([1, 1, 96, 96, 96])

    encoder1 = DynEncoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=6,
        filters = [32, 64, 128, 256, 512, 512],
        strides = [(1, 1, 1)] + [(2, 2, 2) for _ in range(5)],
        kernel_size = [(3, 3, 3) for _ in range(6)],
    )

    encoder2 = DynEncoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=6,
        filters = [32, 64, 128, 256, 512, 512],
        strides = [(1, 1, 1)] + [(2, 2, 2) for _ in range(5)],
        kernel_size = [(3, 3, 3) for _ in range(6)],
    )

    decoder = DynDecoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=6,
        filters = [32, 64, 128, 256, 512, 512],
        strides = [(1, 1, 1)] + [(2, 2, 2) for _ in range(5)],
        kernel_size = [(3, 3, 3) for _ in range(6)],
    )

    print(f'Running the encoder 1')
    output1 = encoder1(input1)
    print(f'to_bottle: ', output1['to_bottle'].shape)
    for i, tensor in enumerate(output1['bridges']):
        print(f'layer {i}: ', tensor.shape)

    print(f'\nRunning the encoder 2')
    output2 = encoder1(input1)
    print(f'to_bottle: ', output2['to_bottle'].shape)
    for i, tensor in enumerate(output2['bridges']):
        print(f'layer {i}: ', tensor.shape)

    print(f'\nRunning decoder')
    output = decoder([output1, output2])
    print('Output: ', output.shape)

    
