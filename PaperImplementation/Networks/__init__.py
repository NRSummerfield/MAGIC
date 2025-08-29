from .nnUNet_wSD import DynUNet_withDualSelfDistillation as nnunet_wsd

from .nnUNet_wSD_decomposed import DynEncoder_wSD, DynDecoder_wSD, DynEncoder, DynDecoder
from .nnUNet_wSD_decomposed_splitOutput import DynDecoder_wSD as SplitDynDecoder_wSD, DynEncoder_wSD as SplitDynEncoder_wSD