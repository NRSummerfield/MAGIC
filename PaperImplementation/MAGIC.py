import os, glob, datetime, math, json, itertools, pickle
from typing import Union, Any, Callable, Optional, Sequence

# PyTorch
import torch
from torch.nn import Module
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cudnn
cudnn.benchmark = True
from torch.optim import Optimizer
from torch_optimizer import Lookahead

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

# Torchmanager
from torchmanager_core.view import logger
from torchmanager import callbacks, losses
from torchmanager_core.protocols import MonitorType

# Other
import numpy as np

# Local
from Networks import DynEncoder_wSD, DynDecoder_wSD

class post_proc_options:
    NONE = 0
    ARGMAX_SCALE_TO_ONE = 1
    ARGMAX_TO_ONE_HOT = 2
    ARGMAX = 3

def skip_call(x: torch.Tensor) -> torch.Tensor:
    return x

def argmax_scale_to_one(x: torch.Tensor, dim=1) -> torch.Tensor:
    return torch.argmax(x, dim=dim, keepdim=True) / (x.shape[dim] - 1)
                                                     
def argmax(x: torch.Tensor, dim=1) -> torch.Tensor:
    return torch.argmax(x, dim=dim, keepdim=True)

def argmax_to_onehot_noBKH(x: torch.Tensor, dim=1) -> torch.Tensor:
    _slice = tuple([slice(None) if d != dim else slice(1, None) for d in range(dim + 1)])
    return one_hot(torch.argmax(x, dim=dim, keepdim=True), num_classes=x.shape[dim], dim=dim)[_slice]

post_processing: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {
    0: skip_call,
    1: argmax_scale_to_one,
    2: argmax_to_onehot_noBKH,
    3: argmax
}

class optimizer_level:
    BRANCH = 'branch'
    CROSS_BRANCH = 'cross_branch'
    TREE = 'tree'
    FOREST = 'forest'

class grouped_iteration:
    def __init__(self, total: int, grouped_lengths: list[int] = [], start: int = 0, step: int = 1):
        self.total = total
        self.config = dict(total=total, grouped_lengths=grouped_lengths, start=start, step=step)

        max_idxs = iter(enumerate([sum(grouped_lengths[:(i+1)]) for i in range(len(grouped_lengths))] + [total]))
        grouped_idxs = []
        group_n, step_max = next(max_idxs)
        for i in range(start, total, step):
            if i < step_max: grouped_idxs.append((i, group_n))
            else:
                group_n, step_max = next(max_idxs)
                grouped_idxs.append((i, group_n))
        self.grouped_idxs = grouped_idxs

    def __iter__(self):
        return iter(self.grouped_idxs)
    def __len__(self):
        return self.total

class grow_forest(torch.nn.Module):
    def __init__(
            self,
            roi_size: list[int],
            num_cascade_layers: int,
            modality_names: list[str],
            agnostic_name: str,
            num_classes_set: list[int],
            encoder_module: Union[Module, list[Module], tuple[Module, dict], list[tuple[Module, dict]]],
            decoder_module: Union[Module, list[Module], tuple[Module, dict], list[tuple[Module, dict]]],
            optimizer_class: tuple[Optimizer, dict] = None,
            learning_rate_class: tuple[Any, dict] = None,
            loss_functions: dict[str, Callable] = None,
            in_channels_key: Union[str, list[str]] = 'in_channels',
            out_channels_key: Union[str, list[str]] = 'out_channels',
            optimizer_connection_level: str = optimizer_level.BRANCH,
            optimize_LookingFoward: Optional[dict] = None,
            optimize_decoder: bool = True,
            post_processing_option: int = post_proc_options.ARGMAX_SCALE_TO_ONE,
            detach_branch_inputs: bool = False,
            mask_start_level: int = 1,
            mask_input_layers_override: Optional[list[int]] = None
        ):
        super().__init__()
        self.config = dict(
            roi_size=roi_size,
            num_cascade_layers=num_cascade_layers,
            modality_names=modality_names,
            agnostic_name=agnostic_name,
            num_classes_set=num_classes_set,
            encoder_module=encoder_module,
            decoder_module=decoder_module,
            in_channels_key=in_channels_key,
            out_channels_key=out_channels_key,
            optimizer_connection_level=optimizer_connection_level,
            optimize_LookingFoward=optimize_LookingFoward,
            post_processing_option=post_processing_option,
            mask_start_level=mask_start_level,
            mask_input_layers_override=mask_input_layers_override
            )
        self.config_forTraining = dict(
            optimizer_class=optimizer_class,
            learning_rate_class=learning_rate_class,
            loss_functions=loss_functions
            )
        
        self.device = torch.device('cpu')
        self.num_cascade_layers = num_cascade_layers
        self.modality_names = modality_names
        self.agnostic_name = agnostic_name
        self.num_classes_set = num_classes_set
        self.optimizer_class = optimizer_class
        self.optimize_LookingFoward = optimize_LookingFoward
        self.learning_rate_class = learning_rate_class
        self.loss_functions = loss_functions
        self.detach_branch_inputs = detach_branch_inputs
        self.post_processing_option = post_processing_option
        self.mask_start_level = mask_start_level
        self.optimize_decoder = optimize_decoder

        if mask_input_layers_override is None:
            if self.post_processing_option in [1, 3]:
                # If this is set to 1, the output of the post-processing will be a single channel with all classes stacked
                # If this is the case, mask_1 input = 1 channels, mask_2 input = 2 channels
                self.mask_input_layers = [i + 1 for i in range(self.num_cascade_layers - 1)]
            elif self.post_processing_option == 2:
                # If this is set to 2, the output of the post-processing will be multiple binary masks across channels
                # If this is the case, mask_1 input = num_classes[0] - 1 channels, mask_2 input = mask_1 input + num_classes[1] - 1
                # Note, this is to NOT include the background as seperate classes
                num_classes_minus_bkg = [n - 1 for n in self.num_classes_set[:-1]]
                self.mask_input_layers = [sum(num_classes_minus_bkg[:i+1]) for i in range(len(num_classes_minus_bkg))]
            else:
                raise ValueError(f'Only self.post_processing_option coded for options 1 and 2, got {self.post_processing_option}.')
        else: 
            self.mask_input_layers = mask_input_layers_override

        self.in_channels_key = in_channels_key if isinstance(in_channels_key, (list, tuple)) else [in_channels_key for _ in range(num_cascade_layers)]
        self.out_channels_key = out_channels_key if isinstance(out_channels_key, (list, tuple)) else [out_channels_key for _ in range(num_cascade_layers)]

        self.encoder_module = self.QA_module_arg(encoder_module)
        self.decoder_module = self.QA_module_arg(decoder_module)
        
        self.twigs: torch.nn.ModuleDict = self.build_twigs()
        self.branches: dict[str, Callable] = self.build_branches()
        self.trees: dict[str, Callable] = self.build_trees()
        if self.optimizer_class is not None: self.optimizers, self.optimizer_LRSchedulers = self.connect_optimizers(level=optimizer_connection_level)
        else: self.optimizers, self.optimizer_LRSchedulers = None, None

        # Note, Lookahead only wraps the optimizer, so the learning rate scheduler should be attached just to the actual optimizer
        if optimize_LookingFoward is not None and self.optimizers is not None:
            for opt_key in self.optimizers.keys():
                print(f'ATTACHING OPT: {opt_key} to Lookahead {optimize_LookingFoward}')
                self.optimizers[opt_key] = Lookahead(self.optimizers[opt_key], **optimize_LookingFoward)

    def to(self, device = None, *args, **kwargs):
        device = device if device else self.device
        self.device = device
        return super().to(device=device, *args, **kwargs)

    def connect_optimizers(self, level: str = optimizer_level.BRANCH):
        opt, opt_arg = self.optimizer_class
        lrs, lrs_arg = self.learning_rate_class
        optimizers = {}
        optimizer_LRSchedulers = {}

        # Getting modules names
        # Sorting by trees and branches
        trees = {}
        for modality in self.modality_names:
            trees[modality] = {}
            for branch in range(self.num_cascade_layers):

                names = [f'{modality}_Encoder_{branch}']
                if branch >= self.mask_start_level: names.append(f'{self.agnostic_name}_Encoder_{branch}')
                if self.optimize_decoder: names.append(f'Decoder_{branch}')

                trees[modality][branch] = names

        #print(f'{trees=}')

        if level.lower() == "branch": # level "branch" = an optimzier for each branch
            print(f"For {level=}...")
            for tree in trees.keys():
                for branch in trees[tree].keys():
                    opt_key = f'{tree}_{branch}'
                    names = trees[tree][branch]
                    print(f'  {opt_key} = {names}')
                    optimizers[opt_key] = opt(itertools.chain(*[self.twigs[name].parameters() for name in names]), **opt_arg)
                    optimizer_LRSchedulers[opt_key] = lrs(optimizers[opt_key], **lrs_arg)

        elif level.lower() == "tree": # level "tree" = an optimizer for each tree
            for layer in range(self.num_cascade_layers):
                for tree in trees.keys():
                    names = []
                    for branch in range(layer + 1):
                        names.extend(trees[tree][branch])
                        #print(f'{trees[tree][branch]=}')
                    opt_key = f'{tree}_{layer}'
                    #print(f"{opt_key=}, {names=}")
                    optimizers[opt_key] = opt(itertools.chain(*[self.twigs[name].parameters() for name in names]), **opt_arg)
                    optimizer_LRSchedulers[opt_key] = lrs(optimizers[opt_key], **lrs_arg)

        elif level.lower() == "forest":
            print(f'Establishing mode {level}...')
            names = []
            for layer in range(self.num_cascade_layers):
                for tree in trees.keys():
                    for branch in range(layer + 1):
                        for name in trees[tree][branch]:
                            if name not in names:
                                names.append(name)

            print(f'Modules attached: {names}')
            optimizers['opt'] = opt(itertools.chain(*[self.twigs[name].parameters() for name in names]), **opt_arg)
            optimizer_LRSchedulers['opt'] = lrs(optimizers['opt'], **lrs_arg)

        elif level.lower() == "cross_branch":
            print(f'Establishing mode {level}...')
            for layer in range(self.num_cascade_layers):
                names = []
                for tree in trees.keys():
                    for name in trees[tree][layer]:
                        if name not in names:
                            names.append(name)
                
                opt_key = f'layer_{layer}'
                optimizers[opt_key] = opt(itertools.chain(*[self.twigs[name].parameters() for name in names]), **opt_arg)
                optimizer_LRSchedulers[opt_key] = lrs(optimizers[opt_key], **lrs_arg)
                print(f'  For Optimizer {opt_key}, included twigs are: {names}')

        else: raise ValueError(f'{level=} not found in accepted ["trees", "branch", "forest", "cross_branch"]')

        return optimizers, optimizer_LRSchedulers

    def build_twigs(self) -> torch.ModuleDict:
        modules = {}
        # Adding in the names of the input modalities
        for modality in self.modality_names:
            for level, (module, args) in enumerate(self.encoder_module):
                args[self.in_channels_key[level]] = 1
                args[self.out_channels_key[level]] = self.num_classes_set[level]
                modules[f'{modality}_Encoder_{level}'] = module(**args)

        # Adding in agnostic input for the labels
        # The MASK_ENCODER is 0 for layer 0, 1 w/ 1 input for layer 1, 2 w/ 2 input for layer 2
        # ADDED IN THE OPTION TO MAKE THE MASK_ENCODER START AT 0
        for level, (module, args) in enumerate(self.encoder_module[self.mask_start_level:]):
            args[self.in_channels_key[level + self.mask_start_level]] = self.mask_input_layers[level] # note, the self.mask_input_layers are planned with respect to level already (i.e. i = 0 == level = 1).
            print(level, self.mask_input_layers[level])
            args[self.out_channels_key[level + self.mask_start_level]] = self.num_classes_set[level + self.mask_start_level]
            modules[f'{self.agnostic_name}_Encoder_{level + self.mask_start_level}'] = module(**args)

        # Adding in the decoders
        # The decoder does not need to have a consistent output because there is only itself to worry about
        for level, (model, args) in enumerate(self.decoder_module):
            args[self.in_channels_key[level]] = 1
            args[self.out_channels_key[level]] = self.num_classes_set[level]
            modules[f'Decoder_{level}'] = model(**args)

        return torch.nn.ModuleDict(modules)        
    
    def build_branches(self):
        branches = {}
        # Iterating through the modality names and layers
        for modality in self.modality_names:
            for level in range(self.num_cascade_layers):
                
                # Retrieving the modules in a specific order
                encoders = [self.twigs[f'{modality}_Encoder_{level}']]
                if level >= self.mask_start_level: encoders.append(self.twigs[f'{self.agnostic_name}_Encoder_{level}'])
                decoder = self.twigs[f'Decoder_{level}']

                # Defining a callable object that iterates through the encoders and decoder
                def wrapped_model(x: torch.Tensor, for_training: bool = False, encoders=encoders, decoder=decoder) -> torch.Tensor:
                    bottle_necks = [encoders[0](x[:, [0]])]
                    idxs = [i for i in range(x.shape[1]) if i != 0]
                    if idxs: bottle_necks.append(encoders[1](x[:, idxs]))

                    if not for_training: return decoder(bottle_necks)
                    else: return decoder(bottle_necks), bottle_necks
                
                branches[f'{modality}_Model_{level}'] = wrapped_model
        
        return branches
    
    def build_trees(self):
        trees = {}
        
        for layer in range(self.num_cascade_layers):            
    
            for modality in self.modality_names:
                _models = []

                for l in range(layer + 1):
                    _models.append(self.branches[f'{modality}_Model_{l}'])
            
                def model(x: torch.Tensor, for_training: bool = False, _num_layers=layer+1, _models=_models) -> torch.Tensor:
                    inputs = [x]
                    outputs = []

                    for i in range(_num_layers):
                        output = _models[i](torch.concat(inputs, dim=1), for_training=for_training)
                        if not for_training and i != _num_layers - 1: 
                            inputs.append(post_processing[self.post_processing_option](output['out']))
                        elif for_training and i != _num_layers - 1: 
                            layer_out = post_processing[self.post_processing_option](output[0]['out'])
                            if self.detach_branch_inputs: layer_out = layer_out.detach()
                            inputs.append(layer_out)

                        outputs.append(output)
                    
                    if not for_training: return torch.concat([output['out'] for output in outputs], dim=1)
                    else: return outputs

                trees[f'{modality}_{layer}'] = model
        return trees
    
    def save_forest(self, dst: str, save_weights_only: bool = False, save_for_continued_training: bool = False, info: dict = {}, training_info: dict = {}):
        os.makedirs(dst, exist_ok=True)
        info['config'] = self.config
        training_info['config'] = self.config_forTraining
        info['save_for_continued_training'] = save_for_continued_training

        # Saving the models
        module_dst = os.path.join(dst, 'Modules')
        os.makedirs(module_dst, exist_ok=True)
        info['module_paths'] = {}
        info['module_Weights_only'] = save_weights_only
        for twig in self.twigs.keys():
            obj_to_save = self.twigs[twig].state_dict() if save_weights_only else self.twigs[twig]
            save_name = f'{twig}_WEIGHTS.pth' if save_weights_only else f'{twig}.pth'
            _dst = os.path.join(module_dst, save_name)
            torch.save(obj_to_save, _dst)
            info['module_paths'][twig] = _dst
        
        if save_for_continued_training:
            # Saving the optimizers
            opt_dst = os.path.join(dst, 'Optimizers')
            os.makedirs(opt_dst, exist_ok=True)
            training_info['optimizers'] = {}
            training_info['optimizer_LRSchedulers'] = {}
            for opt in self.optimizers.keys():
                obj_to_save = self.optimizers[opt]
                save_name = f'{opt}_Optimizer.pth'
                _dst = os.path.join(opt_dst, save_name)
                torch.save(obj_to_save, _dst)
                training_info['optimizers'][opt] = _dst

                obj_to_save = self.optimizer_LRSchedulers[opt]
                save_name = f'{opt}_LRScheduler.pth'
                _dst = os.path.join(opt_dst, save_name)
                torch.save(obj_to_save, _dst)
                training_info['optimizer_LRSchedulers'][opt] = _dst

            # Saving the loss functions
            lossfn_dst = os.path.join(dst, 'LossFunctions')
            os.makedirs(lossfn_dst, exist_ok=True)
            training_info['loss_fns'] = {}
            for lfn in self.loss_functions.keys():
                obj_to_save = self.loss_functions[lfn]
                save_name = f'{lfn}_LossFn.pth'
                _dst = os.path.join(lossfn_dst, save_name)
                torch.save(obj_to_save, _dst)
                training_info['loss_fns'][opt] = _dst

            # Save the info
            with open(os.path.join(dst, 'saved_items_forTraining.pickle'), 'wb') as f:
                pickle.dump(training_info, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save the info
        with open(os.path.join(dst, 'saved_items.pickle'), 'wb') as f:
            pickle.dump(info, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_layer(self, layers: Union[int, Sequence[int]], mst_path: str, pickle_name:str = "saved_items.pickle", cout: Callable = print, reconnect_optimizers: Optional[str] = None):
        layers = layers if isinstance(layers, (tuple, list)) else [layers]

        with open(os.path.join(mst_path, pickle_name), 'rb') as f:
            info: dict = pickle.load(f)

        if _epoch := info.get('current_epoch', None):
            cout(f'Loading layer(s) {layers} from epoch {_epoch}...')

        # Load in the saved twigs
        if info['module_Weights_only']:
            for twig in info['module_paths'].keys():
                # Only loading the twig if it belongs to the layers of interest
                if True in [f'_{layer}' in twig for layer in layers]:
                    state_dict = torch.load(info['module_paths'][twig], map_location=self.device)
                    self.twigs[twig].load_state_dict(state_dict)
        else:
            for twig in info['module_paths'].keys():
                # Only loading the twig if it belongs to the layers of interest 
                if True in [f'_{layer}' in twig for layer in layers]:
                    self.twigs[twig] = torch.load(info['module_paths'][twig], map_location=self.device)

        if reconnect_optimizers is not None:
            self.optimizers, self.optimizer_LRSchedulers = self.connect_optimizers(level=reconnect_optimizers)


    @classmethod
    def load_forest(cls, mst_path: str, pickle_name:str = "saved_items.pickle", pickle_name_forTraining: Optional[str] = None):
        print(f'Loading model info...')
        with open(os.path.join(mst_path, pickle_name), 'rb') as f:
            info = pickle.load(f)
        config = info['config']
        
        if pickle_name_forTraining is not None:
            print(f'Loading [Optional] training info...')
            with open(os.path.join(mst_path, pickle_name_forTraining), 'rb') as f:
                training_info = pickle.load(f)
            config = config | training_info['config']

        # initialize an equivalent raw model
        forest = cls(**config)

        print(f'Loading the models...')
        # Load in the saved twigs
        if info['module_Weights_only']:
            for twig in info['module_paths'].keys():
                state_dict = torch.load(info['module_paths'][twig], map_location=torch.device('cpu'))
                forest.twigs[twig].load_state_dict(state_dict)
        else:
            twigs = {}
            for twig in info['module_paths'].keys():
                twigs[twig] = torch.load(info['module_paths'][twig], map_location=torch.device('cpu'))
            forest.twigs = torch.nn.ModuleDict(twigs)

        if pickle_name_forTraining is not None:

            print(f'Loading the optimizer...')
            # Load in the saved optimizers
            optimizers = {}
            optimizer_paths = sorted(glob.glob(os.path.join(mst_path, 'Optimizers', '*_Optimizer.pth')))
            for path in optimizer_paths:
                opt = os.path.split(path)[-1].split('_Optimizer.pth')[0]
                optimizers[opt] = torch.load(path, map_location=torch.device('cpu'))
            forest.optimizers = optimizers

            print(f'Loading learning rate schedulers...')
            # Load in the saved learning rate schedulers
            optimizer_LRSchedulers = {}
            optimizer_LRScheduler_paths = sorted(glob.glob(os.path.join(mst_path, 'Optimizers', '*_LRScheduler.pth')))
            for path in optimizer_LRScheduler_paths:
                opt = os.path.split(path)[-1].split('_LRScheduler.pth')[0]
                optimizer_LRSchedulers[opt] = torch.load(path, map_location=torch.device('cpu'))
            forest.optimizer_LRSchedulers = optimizer_LRSchedulers

            print(f'Loading loss functions...')
            # Load in the saved loss functions
            loss_functions = {}
            loss_fn_paths = sorted(glob.glob(os.path.join(mst_path, 'LossFunctions', '*_LossFn.pth')))
            for path in loss_fn_paths:
                lfn = os.path.split(path)[-1].split('_LossFn.pth')[0]
                loss_functions[lfn] = torch.load(path, map_location=torch.device('cpu'))
            forest.loss_functions = loss_functions
    
        print(f'Done.')
        return forest

    def QA_module_arg(self, module_arg):
        # QA the encoder input
        check = False
        # Check to see if the encoder is just a module
        if not isinstance(module_arg, (list, tuple)):
            module_arg = [(module_arg, {}) for _ in range(self.num_cascade_layers)]
            check = True
        # Check to see if the encoder is a list
        elif isinstance(module_arg, (list, tuple)):
            # Check to see if it is a list of modules
            if False not in [not isinstance(enc_mod, (list, tuple, dict)) for enc_mod in module_arg]:
                module_arg = [(enc_mod, {}) for enc_mod in module_arg]
                check = True
            # Check to see if it is just a (Module, Args) input
            elif len(module_arg) == 2:
                if False not in [not isinstance(module_arg[0], (list, tuple)), isinstance(module_arg[1], dict)]:
                    module_arg = [module_arg for _ in range(self.num_cascade_layers)]
                    check = True
            # Check to see if it is a list of tuples (i.e (Module, Args))
            elif False not in [isinstance(enc_mod, (tuple, list)) for enc_mod in module_arg]:
                check = True
        if len(module_arg) != self.num_cascade_layers: check = False
        arg_name = f'{module_arg=}'.split('=')[0]
        assert check, f'Argument {arg_name} failed, check input.'
        return module_arg

def format_training_tree_output(tree_output: list[tuple[dict, list[dict]]]):
    output = {}
    for model_level in range(len(tree_output)):
        output[model_level] = tree_output[model_level][0]['out']
    return output
    
def format_training_wSD_tree_output(tree_output: list[tuple[dict, list[dict]]]):
    """
    Formats an output of a multi-branch tree for use in a loss function

    Expected input format:
        list - One element for each layer in the tree
            tuple - One element for decoder / encoder(s)
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
        dict - one element for each layer
            N: dict - output of a combined branch
                'out': torch.Tensor
                'decoder': {
                    'teacher': torch.Tensor
                    'students': list[torch.Tensor]
                    }
                'Encoder_N': {
                    'teacher': torch.Tensor
                    'students': list[torch.Tensor]
                    }
    """
    output = {}
    for model_level in range(len(tree_output)):
        output[model_level] = {}
        decoder_output, encoder_outputs = tree_output[model_level]

        output[model_level]['out'] = decoder_output['out']
        for i, encoder_output in enumerate(encoder_outputs): output[model_level][f'encoder_{i}'] = {'students': encoder_output['students'], 'teacher': encoder_output['teacher']}
        output[model_level]['decoder'] = {'students': decoder_output['students'], 'teacher': decoder_output['teacher']}

    return output

def argmax_with_multiOutput(x: torch.Tensor, num_classes_set: list[int]) -> torch.Tensor:
    outs = []
    start_index = 0
    for i, n in enumerate(num_classes_set):
        post_proc = AsDiscrete(argmax=True, to_onehot=n)
        end_index = start_index + n
    
        out = post_proc(x[start_index:end_index])[1:]
        start_index = end_index
        outs.append(out)
    return torch.concat(outs, dim=0)

class lr_lambda:
    def __init__(self, LRS_converge_epoch, gamma):
        self.lrs_ce = LRS_converge_epoch
        self.g = gamma
    def __call__(self, epoch):
        return (1 - epoch / self.lrs_ce) ** self.g
    
if __name__ == '__main__':
    from torch.optim import AdamW
    from monai.inferers.utils import sliding_window_inference


    model_hyperparams = dict(
        scale_factors=[1, 2, 4, 8, 16],
        spatial_dims=3,
        filters = [32, 64, 128, 256, 512, 512],
        strides = [(1, 1, 1)] + [(2, 2, 2) for _ in range(5)],
        kernel_size = [(3, 3, 3) for _ in range(6)],
    ) 

    # ^ predefined things
    # **************
    # v input arguments

    num_cascade_layers: int = 3

    modality_names: list[str] = ['VR', 'CCTA']
    agnostic_name: str = 'Mask'
    num_classes_set: list[int] = [2, 10, 4]
    forest = grow_forest(
        num_cascade_layers=3,
        modality_names=['VR', 'CCTA', 'simCT'],
        agnostic_name='Mask',
        num_classes_set=[2, 10, 4],
        encoder_module=(DynEncoder_wSD, model_hyperparams),
        decoder_module=(DynDecoder_wSD, model_hyperparams),
        optimizer_class=(AdamW, dict(lr=1e-3, weight_decay=1e-5)),
        learning_rate_class=(torch.optim.lr_scheduler.LambdaLR, dict(lr_lambda=lr_lambda(1000, gamma=0.9))),
        loss_functions={},
        in_channels_key='in_channels',
        out_channels_key='out_channels',
        optimizer_connection_level=optimizer_level.TREE,
    )

    forest.save_forest('TESTING_FOREST', save_weights_only=False)

    print(f'Branch keys:')
    print(f'  ', forest.branches.keys())
    print(f'Optimizer keys:')
    print(f'  ', forest.optimizers.keys())
    print(f'Twig keys:')
    print(f'  ', forest.twigs.keys())

    print()
    forest.load_forest('TESTING_FOREST')

    print(f'Branch keys:')
    print(f'  ', forest.branches.keys())
    print(f'Optimizer keys:')
    print(f'  ', forest.optimizers.keys())
    print(f'Twig keys:')
    print(f'  ', forest.twigs.keys())

    device = torch.device('cpu')
    input = torch.rand([1, 1, 96, 96, 96]).to(device)
    forest = forest.to(device)

    # print(f'\nUnit testing the input / output')
    # print(f'Validation / Testing cases:')
    # print(f'Input of shape {input.shape} on device {input.device}')
    # validation_output = forest.trees[forest.modality_names[0]](input)

    # print(type(validation_output))
    # print(validation_output.shape)

    # print(f'Training cases:')
    # print(f'Input of shape {input.shape} on device {input.device}')
    # training_output = forest.trees[forest.modality_names[0]](input, for_training=True)

    # training_output = format_training_wSD_tree_output(training_output)
    # for k in training_output.keys():
    #     print(f'  {k=}')
    #     for kk in training_output[k].keys():
    #         print(f'    {kk=}')

    print(f'Sliding window testing case:')
    input = torch.rand([1, 1, 128, 128, 128]).to(device)
    testing_output = sliding_window_inference(inputs=input, roi_size=(96, 96, 96), sw_batch_size=1, predictor=forest.trees[forest.modality_names[0]])
    print(testing_output.shape)
    out = argmax_with_multiOutput(testing_output[0], forest.num_classes_set)
    print(out.shape)


