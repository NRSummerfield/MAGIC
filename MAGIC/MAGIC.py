import os, glob, datetime, math, json, itertools, pickle
from typing import Union, Any, Callable, Optional, Sequence
from functools import partial

# PyTorch
import torch
from torch.nn import Module
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cudnn
cudnn.benchmark = True
from torch.optim import Optimizer
from torch_optimizer import Lookahead
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn import DataParallel

# Monai
import monai
from monai.data.dataset import Dataset
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.data.utils import pad_list_data_collate
from monai.losses.dice import DiceCELoss
from monai.data.dataloader import DataLoader
from monai.inferers.utils import sliding_window_inference
import monai.transforms as transform
from monai.transforms import AsDiscrete
from monai.networks.utils import one_hot

# Other
import numpy as np

# Local
from Networks import DynEncoder_wSD, DynDecoder_wSD

class stepper:
    """Placeholder class that has a .step() method that does nothing"""
    def __init__(self):
        self.flagged = False
    def step(self):
        if not self.flagged:
            print(f'[WARNING] A learning rate scheduler is not implemented')
            self.flagged = True


def out_caller(input: dict) -> torch.Tensor:
    """My implimentation uses an output dictionary with the final prediction corresponding with key 'out'"""
    return input['out']

def default_caller(input: Any) -> Any:
    """Default callable option that does nothing and returns the input"""
    return input

def QA_arg_input(object: Any) -> Union[partial, Any]:
    """Simple function to apply input arguments to a function for later use"""
    if isinstance(object, (tuple, list)): return partial(object[0], **object[1])
    else: return object

class MAGIC_model(torch.nn.Module):
    def __init__(self,
            modality_names: list[str],
            agnostic_name: str,
            encoder_module: Union[Module, tuple[Module, dict], partial[Module]],
            decoder_module: Union[Module, tuple[Module, dict], partial[Module]],
            out_caller: Callable[[Any], torch.Tensor] = default_caller,
                 ):
        super().__init__()
        self.target = modality_names[0]
        self.modality_names = modality_names
        self.agnostic_name = agnostic_name

        self.encoder_module = QA_arg_input(encoder_module)
        self.decoder_module = QA_arg_input(decoder_module)
        self.out_caller = out_caller

        self.magic_modules = self.build_modules()

    def set_target(self, target: str):
        self.target = target
    
    def build_modules(self):
        """
        Builds the modules needed for MAGIC:
        1 encoder per modality & 1 decoder
        """
        modules = {}
        # Iterating through the modality-specific layers
        for modality in self.modality_names:
            modules[modality] = self.encoder_module()
        
        # Adding the modality-agnostic decoder
        modules[self.agnostic_name] = self.decoder_module()

        return torch.nn.ModuleDict(modules)
    
    def forward(self, x: torch.Tensor, for_training: bool = False) -> torch.Tensor:
        encoder_out = self.magic_modules[self.target](x)
        if for_training: return self.magic_modules[self.agnostic_name](encoder_out), encoder_out
        else: return self.out_caller(self.magic_modules[self.agnostic_name](encoder_out))
    
    def __getitem__(self, target:str) -> torch.nn.Module:
        self.set_target(target)
        return self.forward
        

class MAGIC_framework(torch.nn.Module):
    def __init__(
            self,
            roi_size: list[int],
            modality_names: list[str],
            agnostic_name: str,
            encoder_module: Union[Module, tuple[Module, dict], partial[Module]],
            decoder_module: Union[Module, tuple[Module, dict], partial[Module]],
            out_caller: Callable[[Any], torch.Tensor] = default_caller,
            optimizer_class: Union[Optimizer, tuple[Optimizer, dict], partial[Optimizer]] = None,
            learning_rate_class: Union[Any, tuple[Any, dict], partial[Any]] = None,
            loss_function: Callable = None,
        ):
        """
        The MAGIC class to facilitate modality-agnostic and overlapping segmentation

        ---
        Args:
        * roi_size: 
        * modality_names: 
        * agnostic_name: 
        * encoder_module: 
        * decoder_module: 
        * out_caller: 
        * optimizer_class: 
        * learning_rate_class: 
        * loss_function:  
        """
        super().__init__()

        # Tracking the input parameters
        self.config = dict(
            roi_size=roi_size,
            modality_names=modality_names,
            agnostic_name=agnostic_name,
            encoder_module=encoder_module,
            decoder_module=decoder_module,
            out_caller=out_caller,
            )
        self.config_forTraining = dict(
            optimizer_class=optimizer_class,
            learning_rate_class=learning_rate_class,
            loss_function=loss_function,
            )
        
        self.device = torch.device('cpu')
        self.modality_names = modality_names
        self.agnostic_name = agnostic_name
        self.out_caller = out_caller
        self.loss_function = loss_function

        # Re-check this implementation - May be redundant
        self.optimizer_class = QA_arg_input(optimizer_class)
        self.learning_rate_class = QA_arg_input(learning_rate_class)

        self.tricks = MAGIC_model(
            modality_names=modality_names,
            agnostic_name=agnostic_name,
            encoder_module=encoder_module,
            decoder_module=decoder_module,
            out_caller=out_caller
        )
        self.optimizer, self.optimizer_LRS = self.connect_optimizer()

    def set_target(self, target:str):
        if isinstance(self.tricks, (DistributedDataParallel, DataParallel)): return self.tricks.module.set_target(target)
        else: return self.tricks.set_target(target)
    
    def __getitem__(self, target:str):
        return self.tricks[target]

    def connect_optimizer(self) -> Union[tuple[None, None], tuple[torch.optim.Optimizer, stepper], tuple[torch.optim.Optimizer, Any]]:
        """
        [Optional] Connects all the modules in MAGIC to a single optimizer
            -> If no optimizer is provided, both `self.optimizer` and `self.optimizer_lrs` are None
        [Optional] Connects the optimizer to a learning rate scheduler
            -> If an optimizer is provided both not a learning rate scheduler, the `self.optimizer_lrs` is replaced with `stepper` that contains a `.step()` method that does nothing for clean integration
        """
        if self.optimizer_class is None: return None, None
        optimizer = self.optimizer_class(itertools.chain(*[self.tricks.magic_modules[k].parameters() for k in self.tricks.magic_modules.keys()]))
        if self.learning_rate_class is None: return optimizer, stepper()
        opt_lrs = self.learning_rate_class(optimizer)
        return optimizer, opt_lrs

    def save(self, dst: str, save_weights_only: bool = True, save_for_training: bool = False, info: dict = {}, training_info: dict = {}):
        """
        Method to save MAGIC. Pickles the input parameters and saves the modules as either torch weights or a torch module

        NOTE: `save_weights_only` is recommended to be False
        """
        if isinstance(self.tricks, (DataParallel, DistributedDataParallel)):
            tricks = self.tricks.module
        else:
            tricks = self.tricks

        os.makedirs(dst, exist_ok=True)
        info['config'] = self.config
        training_info['config'] = self.config_forTraining
        info['save_for_training'] = save_for_training

        # Saving the modules
        module_dst = os.path.join(dst, 'Modules')
        os.makedirs(module_dst, exist_ok=True)
        info['module_paths'] = {}
        info['module_weights_only'] = save_weights_only
        for module in tricks.magic_modules.keys():
            obj_to_save = tricks.magic_modules[module]
            if save_weights_only: obj_to_save = obj_to_save.state_dict()
            
            save_name = f'{module}_WEIGHTS.pth' if save_weights_only else f'{module}.pth'
            _dst = os.path.join(module_dst, save_name)
            torch.save(obj_to_save, _dst)
            info['module_paths'][module] = _dst

        if save_for_training:
            train_dst = os.path.join(dst, 'TrainingObjects')
            os.makedirs(train_dst, exist_ok=True)
            # Saving the optimizers
            training_info['optimizer_is_None'] = True
            if self.optimizer is not None:
                obj_to_save = self.optimizer
                save_name = f'Optimizer.pth'
                _dst = os.path.join(train_dst, save_name)
                torch.save(obj_to_save, _dst)
                training_info['optimizer'] = _dst
                training_info['optimizer_is_None'] = False

                training_info['lrs_is_None'] = True
                if not isinstance(self.optimizer_LRS, stepper):
                    obj_to_save = self.optimizer_LRS
                    save_name = f'LRScheduler.pth'
                    _dst = os.path.join(train_dst, save_name)
                    torch.save(obj_to_save, _dst)
                    training_info['optimizer_LRScheduler'] = _dst
                    training_info['lrs_is_None'] = False

            # Saving the loss functions
            training_info['lrs_is_None'] = True
            if self.loss_function is not None:
                lossfn_dst = os.path.join(dst, 'LossFunctions')
                os.makedirs(lossfn_dst, exist_ok=True)
                training_info['loss_fns'] = {}
                obj_to_save = self.loss_function
                save_name = f'LossFn.pth'
                _dst = os.path.join(lossfn_dst, save_name)
                torch.save(obj_to_save, _dst)
                training_info['loss_fn'] = _dst
                training_info['lrs_is_None'] = False

            with open(os.path.join(dst, 'training_items.pickle'), 'wb') as f:
                pickle.dump(training_info, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(dst, 'saved_items.pickle'), 'wb') as f:
            pickle.dump(info, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_magic(cls, mst_path: str, pickle_name:str = "saved_items.pickle", pickle_name_forTraining: Optional[str] = None, cout: Callable = print):
        cout(f'Loading model info from {mst_path} > {pickle_name}...')
        # Load the base-required info 
        with open(os.path.join(mst_path, pickle_name), 'rb') as f:
            info = pickle.load(f)
        config = info['config']
        
        # Look to see if training information was saved
        if pickle_name_forTraining is not None:
            cout(f'Loading [Optional] training info...')
            with open(os.path.join(mst_path, pickle_name_forTraining), 'rb') as f:
                training_info = pickle.load(f)
            config = config | training_info['config']
        
        # Initialize based off of saved config sats
        magic = cls(**config)

        cout(f'Loading the models...')
        if info['module_weights_only']:
            # If it is only the weights that need to be loaded, load them into the initialized modules
            for twig in info['module_paths'].keys():
                state_dict = torch.load(info['module_paths'][twig], map_location=torch.device('cpu'))
                magic.tricks.magic_modules[twig].load_state_dict(state_dict)
        else:
            # Directly replace the moduels with loaded modules otherwise [Not recommended]
            twigs = {}
            for twig in info['module_paths'].keys():
                twigs[twig] = torch.load(info['module_paths'][twig], map_location=torch.device('cpu'))
            magic.tricks.magic_modules = torch.nn.ModuleDict(twigs)
            # Reconnect the optimizers onces the new modules are loaded
            magic.optimizers, magic.optimizer_LRSchedulers = magic.connect_optimizer()

        if pickle_name_forTraining is not None:
            raise NotImplementedError
        
        cout(f'Done.')
        return magic

    def load_module(self, module_names: Union[str, list[str]], mst_path: str, pickle_name:str = "saved_items.pickle", cout: Callable = print, reconnect_optimizers: Optional[str] = None):
        module_names = module_names if isinstance(module_names, (tuple, list)) else [module_names]

        cout(f'Loading select modules from {mst_path} > {pickle_name}')
        with open(os.path.join(mst_path, pickle_name), 'rb') as f:
            info: dict = pickle.load(f)

        if _epoch := info.get('current_epoch', -1):
            cout(f'  Loading module(s) {module_names} from epoch {_epoch}...')

        # Load in the saved modules
        if info['module_weights_only']:
            for module_name in module_names:
                state_dict = torch.load(info['module_paths'][module_name], map_location=self.device)
                self.tricks.magic_modules[module_name].load_state_dict(state_dict)

        else:
            for module_name in module_names:    
                self.tricks.magic_modules[module_name] = torch.load(info['module_paths'][module_name], map_location=self.device)

        if reconnect_optimizers is not None:
            self.optimizers, self.optimizer_LRSchedulers = self.connect_optimizer()

    def to(self, device = None, *args, **kwargs):
        device = device if device else self.device
        self.device = device
        return super().to(device=device, *args, **kwargs)
    
    def forward(self, inp: torch.Tensor, for_training: bool = False) -> torch.Tensor:
        return self.tricks(inp, for_training)

