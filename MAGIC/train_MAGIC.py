import os, glob, datetime, math, json, logging, operator
from typing import Union, Iterable, Callable, Any

# PyTorch
import torch
from torch.optim import AdamW
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cudnn
cudnn.benchmark = True

# Monai
import monai
from monai.data.dataset import Dataset
from monai.metrics import DiceMetric, HausdorffDistanceMetric, CumulativeIterationMetric
from monai.data.utils import pad_list_data_collate
from monai.data.dataloader import DataLoader
from monai.inferers.utils import sliding_window_inference
import monai.transforms as transform
from monai.transforms import AsDiscrete

# Other
import numpy as np
from alive_progress import alive_bar

# Local
from Networks import SplitDynEncoder_wSD, SplitDynDecoder_wSD
from joint_dataloader import joint_dataloader
from colored_output import color_options, colorful_print
from Losses import CompositeLoss, PixelWiseKLDiv, MultiGrpLoss, DiceBoundaryDiceEntropyLoss
from MAGIC import MAGIC_framework, out_caller
from MagicianAssistant import GroupSeperate, split_groups_transform, channel_to_stacked_binary_transform, lr_lambda, format_training_output_nnUNetwSD

device = torch.device('cuda:3')

save_dir = f'Experiments/MAGIC_Take2'
os.makedirs(save_dir, exist_ok=True)

# setting up a logger to track the experiment
logging.basicConfig(filename=os.path.join(save_dir, 'Narrative.log'), filemode='a', format='%(asctime)s,%(msecs)d, %(name)s %(levelname)s %(message)s', datefmt="%H:%M:%S", level=logging.INFO)
logging.info("Running training logger")
logger = logging.getLogger('TrainingLog')
def out(*inputs, sep=' ', end='\n', color:color_options = color_options.DEFAULT, underline:bool = False, bold: bool = False):
    logger.info(sep.join([str(x) for x in inputs]) + '')
    colorful_print(*inputs, sep=sep, end=end, color=color, bold=bold, underline=underline)
cout = out

description = """
Running MAGIC post-changes
"""
cout(description)

# Defining the output group idxs
label_idxs=[ # groups WITH OUT background
    [0], # WH
    [1, 2, 3, 4, 5, 6, 7, 8, 9], # Chambers + Great Vessels
    [10, 11, 12, 13, 14, 15, 16, 17], # Coronary Arteries + Valves
    [18, 19] # Nodes
    ]
grp_idxs = [ # groups WITH background
    [0, 1], # BKG + WH
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # BKG + Chambers + Great Vessels
    [12, 13, 14, 15, 16, 17, 18, 19, 20], # BKG + Coronary Arteries + Valves
    [21, 22, 23], # BKG + Nodes
    ]
    
# ----------------------------------------------------------------
# Pre-processing / Augmentation
# ----------------------------------------------------------------
image_keys = ['image']
real_keys = image_keys + ['label']
all_keys = image_keys + [f'grp_{l}' for l in range(len(label_idxs))]

preprocessing_transforms = [
    transform.LoadImaged(real_keys),
    split_groups_transform(
        target_key='label',
        group_idxs=label_idxs,
        out_names=[f'grp_{l}' for l in range(len(label_idxs))],
        ),
    channel_to_stacked_binary_transform('label', 'blabel'),
    transform.NormalizeIntensityd(image_keys, nonzero=True), #z-score normalization that helps consistancy with patient to patient and brings mean to zero to help with deep learning
]

roi_size = (96, 96, 96)
range_10 = [math.radians(-10), math.radians(10)]
augmentation_transforms = [
    transform.RandCropByPosNegLabeld(all_keys, image_key = image_keys[0], label_key = 'blabel', neg = 1, spatial_size = roi_size, num_samples = 2),
    transform.RandRotate90d(all_keys, prob = 0.5, max_k = 3),
    transform.RandRotated(all_keys, range_x = range_10, range_y = range_10, range_z = range_10, mode = ['bilinear'] + ['nearest']*len(label_idxs)),
    transform.RandFlipd(all_keys, spatial_axis = [0,1,2], prob = 0.5),
    transform.RandScaleIntensityd(image_keys, prob = 1, factors = 0.1),
    transform.RandShiftIntensityd(image_keys, offsets = 0.1, prob = 1)
]

batch_size = 1

# ----------------------------------------------------------------
# Data setup
# ----------------------------------------------------------------

master_src = "path/to/data/directory"
# The implemented directory is configured in the following way:

# Data directory:
#   | Modality 1
#       | Institution 1
#           | Image & Label w/ PID .nii.gz
#           | ...
#       | Institution 2
#           | Image & Label w/ PID .nii.gz
#           | ...
#   | Modality 2
#       | Institution 1
#           | Image & Label w/ PID .nii.gz
#           | ...
#       | Institution 2
#           | Image & Label w/ PID .nii.gz
#           | ...
#   | ...

# the following lines of code define what [deidentified] PIDs correspond to what training / valdiation / testing splits (pre-randomized and hard coded)

# NOTE: This includes prior examed psuedo labels for semi-supervised training of MAGIC. To run step 1 (i.e. fully-supervised), remove the pseudo labels and run on only hard labeled images

# ================================
# VR data
# ================================

main_path = os.path.join(master_src, "VR")

# HFHS
HF_training_pids = [5, 8, 10, 11, 17, 21, 27, 31, 34, 40]
HF_validation_pids = [6, 20, 39]
HF_testing_pids = [1, 18, 29, 35, 36]

training_pids = [f'HF_VR_{pid:02d}' for pid in HF_training_pids]
validation_pids = [f'HF_VR_{pid:02d}' for pid in HF_validation_pids]
testing_pids = [f'HF_VR_{pid:02d}' for pid in HF_testing_pids]

# UW
UW_training_pids = [11, 13, 15]
UW_validation_pids = [12, 14]
UW_testing_pids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

training_pids += [f'UW_VR_{pid:02d}' for pid in UW_training_pids]
validation_pids += [f'UW_VR_{pid:02d}' for pid in UW_validation_pids]
testing_pids += [f'UW_VR_{pid:02d}' for pid in UW_testing_pids]

# Carefully curated pseudo labels for training only
HF_pseudo_training_pids = ['05_FX2', '08_FX2', '10_FX2', '10_FX4', '17_FX2', '17_FX3', '21_FX2', '21_FX3', '27_FX3', '31_FX2', '31_FX3', '34_FX2', '34_FX3', '40_FX2', '40_FX3']
training_pids += [f'HFPred_VR_{pid}' for pid in HF_pseudo_training_pids]
UW_pseudo_training_pids = ['11_FX2', '11_FX3', '11_FX4', '13_FX2', '13_FX3', '13_FX4', '15_FX3', '16_SIM', '16_FX3', '16_FX4']
training_pids += [f'UWPred_VR_{pid}' for pid in UW_pseudo_training_pids]

validation_data = []
training_data = []
testing_data = []

for pid in validation_pids:
    image_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}_SIM.IMAGE.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}_SIM.LABEL.nii.gz")))
    for i in range (len(image_paths)):
        validation_data.append({'image': image_paths[i], 'label': label_paths[i]})

for pid in testing_pids:
    image_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}_SIM.IMAGE.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}_SIM.LABEL.nii.gz")))
    for i in range (len(image_paths)):
        testing_data.append({'image': image_paths[i], 'label': label_paths[i]})

for pid in training_pids:
    image_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}*IMAGE.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}*LABEL.nii.gz")))
    for i in range (len(image_paths)):
        training_data.append({'image': image_paths[i], 'label': label_paths[i]})

training_dataset = Dataset(training_data, transform.Compose(preprocessing_transforms+augmentation_transforms))
validation_dataset = Dataset(validation_data, transform.Compose(preprocessing_transforms))
testing_dataset = Dataset(testing_data, transform.Compose(preprocessing_transforms))

VR_training_dataloader = DataLoader(training_dataset,  num_workers = 1, batch_size = batch_size, shuffle = True, collate_fn = pad_list_data_collate, pin_memory = True)
VR_validation_dataloader = DataLoader(validation_dataset,  num_workers = 1, batch_size = 1, collate_fn = pad_list_data_collate, pin_memory = True)
VR_testing_dataloader = DataLoader(testing_dataset,  num_workers = 1, batch_size = 1, collate_fn = pad_list_data_collate, pin_memory = True)


# # ================================
# # CCTA data
# # ================================

main_path = os.path.join(master_src, "CCTA")

# UW
UW_training_pids = [3, 4, 5, 9, 11, 13, 15, 16, 17, 19, 20, 21, 22, 25, 27, 35, 42, 43, 46, 52, 59, 62, 78, 95, 104]
UW_validation_pids = [6, 14, 18, 50, 92]
UW_testing_pids = [3] # PLACEHOLDER FOR REAL TESTING SET

training_pids = [f'UW_CCTA_{pid:03d}' for pid in UW_training_pids]
validation_pids = [f'UW_CCTA_{pid:03d}' for pid in UW_validation_pids]
testing_pids = [f'UW_CCTA_{pid:03d}' for pid in UW_testing_pids]

UW_pseudo_training_pids = [28, 30, 31, 33, 38, 40, 45, 47, 49, 53, 56, 57, 64, 69, 70, 71, 75, 177, 202, 203, 207, 211, 221, 227, 233]
training_pids += [f'UWPred_CCTA_{pid:03d}' for pid in UW_pseudo_training_pids]


validation_data = []
training_data = []
testing_data = []

for pid in validation_pids:
    image_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}.IMAGE.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}.LABEL.nii.gz")))
    for i in range (len(image_paths)):
        validation_data.append({'image': image_paths[i], 'label': label_paths[i]})

for pid in testing_pids:
    image_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}.IMAGE.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}.LABEL.nii.gz")))
    for i in range (len(image_paths)):
        testing_data.append({'image': image_paths[i], 'label': label_paths[i]})

for pid in training_pids:
    image_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}.IMAGE.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}.LABEL.nii.gz")))
    for i in range (len(image_paths)):
        training_data.append({'image': image_paths[i], 'label': label_paths[i]})

training_dataset = Dataset(training_data, transform.Compose(preprocessing_transforms+augmentation_transforms))
validation_dataset = Dataset(validation_data, transform.Compose(preprocessing_transforms))
testing_dataset = Dataset(testing_data, transform.Compose(preprocessing_transforms))

CCTA_training_dataloader = DataLoader(training_dataset,  num_workers = 1, batch_size = batch_size, shuffle = True, collate_fn = pad_list_data_collate, pin_memory = True)
CCTA_validation_dataloader = DataLoader(validation_dataset,  num_workers = 1, batch_size = 1, collate_fn = pad_list_data_collate, pin_memory = True)
CCTA_testing_dataloader = DataLoader(testing_dataset,  num_workers = 1, batch_size = 1, collate_fn = pad_list_data_collate, pin_memory = True)

# # ================================
# # simCT data
# # ================================

main_path = os.path.join(master_src, "simCT")

# HFHS
HF_training_pids = [2, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 21, 24, 26, 30, 33, 35]
HF_validation_pids = [1, 6, 31]
HF_testing_pids = [13, 22, 28, 29, 34]

training_pids = [f'HF_simCT_{pid:02d}' for pid in HF_training_pids]
validation_pids = [f'HF_simCT_{pid:02d}' for pid in HF_validation_pids]
testing_pids = [f'HF_simCT_{pid:02d}' for pid in HF_testing_pids]

# UW
UW_training_pids = [1, 4, 7, 8, 9, 10, 11, 13] # 85, 15, 19, 20, 23, 27, 29, 31, 49, 58, 62, 63, 64, 69, 70, 12, 14, 25, 26, 30
UW_validation_pids = [2, 6]
UW_testing_pids = []

training_pids += [f'UW_simCT_{pid:02d}' for pid in UW_training_pids]
validation_pids += [f'UW_simCT_{pid:02d}' for pid in UW_validation_pids]
testing_pids += [f'UW_simCT_{pid:02d}' for pid in UW_testing_pids]

UW_pseudo_training_pids = [5, 15, 19, 20, 23, 27, 29, 31, 49, 58, 62, 63, 64, 69, 70, 12, 14, 25, 26, 30]
training_pids += [f'UWPred_simCT_{pid:02d}' for pid in UW_pseudo_training_pids]

HF_pseudo_training_pids = [3, 4, 32, 37, 39]
training_pids += [f'HFPred_simCT_{pid:02d}' for pid in HF_pseudo_training_pids]

validation_data = []
training_data = []
testing_data = []

for pid in validation_pids:
    image_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}.IMAGE.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}.LABEL.nii.gz")))
    for i in range (len(image_paths)):
        validation_data.append({'image': image_paths[i], 'label': label_paths[i]})

for pid in testing_pids:
    image_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}.IMAGE.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}.LABEL.nii.gz")))
    for i in range (len(image_paths)):
        testing_data.append({'image': image_paths[i], 'label': label_paths[i]})

for pid in training_pids:
    image_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}.IMAGE.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(main_path, '*', f"{pid}.LABEL.nii.gz")))
    for i in range (len(image_paths)):
        training_data.append({'image': image_paths[i], 'label': label_paths[i]})

training_dataset = Dataset(training_data, transform.Compose(preprocessing_transforms+augmentation_transforms))
validation_dataset = Dataset(validation_data, transform.Compose(preprocessing_transforms))
testing_dataset = Dataset(testing_data, transform.Compose(preprocessing_transforms))

simCT_training_dataloader = DataLoader(training_dataset,  num_workers = 1, batch_size = batch_size, shuffle = True, collate_fn = pad_list_data_collate, pin_memory = True)
simCT_validation_dataloader = DataLoader(validation_dataset,  num_workers = 1, batch_size = 1, collate_fn = pad_list_data_collate, pin_memory = True)
simCT_testing_dataloader = DataLoader(testing_dataset,  num_workers = 1, batch_size = 1, collate_fn = pad_list_data_collate, pin_memory = True)

################################################################
# Full set assembly and QA
################################################################

training_sets: tuple[str, Iterable[torch.Tensor]] = (
    ('VR', VR_training_dataloader),
    ('CCTA', CCTA_training_dataloader),
    ('simCT', simCT_training_dataloader)
)

# Combine all 3 training dataloaders into 1 dataloader that will return the type of the data and path to the data when called
# Randomizes which image type is called when used for training to avoid biasing
training_dataloader = joint_dataloader(
    dataloaders=[VR_training_dataloader, CCTA_training_dataloader, simCT_training_dataloader],
    names = ['VR', 'CCTA', 'simCT'],
    random_iter_choice=True,
    truncate_to_minimum=False,
)
cout(f'There are {len(training_dataloader)} iterations in the joint dataloader...')

# Combine all 3 validation dataloaders such that all data is called upon when running the validation step
validation_sets: tuple[str, Iterable[torch.Tensor]] = (
    ('VR', VR_validation_dataloader),
    ('CCTA', CCTA_validation_dataloader),
    ('simCT', simCT_validation_dataloader),
)

validation_dataloader = joint_dataloader(
    dataloaders=[VR_validation_dataloader, CCTA_validation_dataloader, simCT_validation_dataloader],
    names = ['VR', 'CCTA', 'simCT'],
    random_iter_choice=False,
    truncate_to_minimum=False
)

testing_sets: tuple[str, Iterable[torch.Tensor]] = (
    ('VR', VR_testing_dataloader),
    ('CCTA', CCTA_testing_dataloader),
    ('simCT', simCT_testing_dataloader)
)

# QAing the dataloaders to make sure everything looks correct
for data_set_name, data_set in [("training", training_sets), ("validation", validation_sets), ("testing", testing_sets)]:
    cout(f'QAing the {data_set_name} group...\n')

    for modality, dataloader in data_set:
        cout(f'  Looking at {modality=}... n={len(dataloader)} iterations')
        for ds in dataloader:
            image = ds['image']
            cout(f'    Image shape and range: {image.shape}, {image.min():0.2e} - {image.max():0.2e}')
            for group in [f'grp_{i}' for i in range(len(grp_idxs))]: cout(f'    Label {group} shape and values: {ds[group].shape}, {torch.unique(ds[group])}')
            cout()
            break
    cout('-'*64)

# ================================================================
# ----------------------------------------------------------------
# Model setup
# ----------------------------------------------------------------
# ================================================================
# ----------------------------------------------------------------
# loss_functions
# ----------------------------------------------------------------

# For the loss function weights:
# When you have filters that are 6 elements long (i.e. [32, 64, 128, 256, 512, 512]), the last element is the bottle neck, where no self distillation nor deep supervision is applied
# The remaining 5 represent the encoding / decoding layers where the self distillation and deep supervision impact training
# When running deep supervision, you are comparing the ground truth to the prediction output and the 5 layers as described above. When weighting the loss function, you will therefore have 6 terms.
# When running self distillation, one of the layers per branch (i.e. the most encoded and the most decoded layer) are utilized as the teacher.
# Self distillation is then applied across all remaining layers or students.
# In the current case where you have 5 layers, one forms the teacher, and is compared against the four remanining students.
# As such, you will have 4 weights.

# The segmentation loss is the deep supervision. Each layer of the decoder is output and compared against ground truth using the same metric as the prediction.
# The CompositeLoss function expects in the order of [Prediction, Teacher (most decoded, deepest), Least Decoded (shallowest), ..., Second Most Decoded].
# When deciding the weights here, the prediction should be equivalent to the ground truth, the teacher should be really close to the ground truth, and the shallowest / least decoded is the furthest away.
weights_segmentation_loss = [1.0, 0.4, 0.4, 0.6, 0.8, 1.0] # Something that has work well in the past

# The distillation loss compares the teacher (deepest, most encoded / decoded) branch to the students (the rest) in the order of (shallowest / least encoded, ..., second deepest / second most encoded / decoded)
# When deciding the weights here, the second deepest is expected to be the closest to the teacher and given a higher weight than the others, with the shallowest being furthest away and given a smaller weight.
weights_distillation_loss = [0.4, 0.6, 0.8, 1.0]

# Using a combined loss that examines the output from all 4 groups for the overall loss 
segmentation_loss = MultiGrpLoss
segmentation_loss_args = dict(
    grp_idxs = grp_idxs,
    loss_fns = (
        (DiceBoundaryDiceEntropyLoss, dict(dice_type="Dice", include_background = True, to_onehot_y = True, softmax = True, entropy_type="CrossEntropy")),#, smooth_nr=0.2, smooth_dr=0.2, lambda_e=1)),
        (DiceBoundaryDiceEntropyLoss, dict(dice_type="Dice", include_background = True, to_onehot_y = True, softmax = True, entropy_type="CrossEntropy")),#, smooth_nr=0.2, smooth_dr=0.2, lambda_e=1)),
        (DiceBoundaryDiceEntropyLoss, dict(dice_type="Dice", include_background = True, to_onehot_y = True, softmax = True, gamma=2, entropy_type="Focal")),#, smooth_nr=0.2, smooth_dr=0.2, lambda_e=1)),
        (DiceBoundaryDiceEntropyLoss, dict(dice_type="Dice", include_background = True, to_onehot_y = True, softmax = True, gamma=2, entropy_type="Focal")),#, smooth_nr=0.2, smooth_dr=0.2, lambda_e=1)),
    )
)

# Using a combined loss that exames the self-distillation from all 4 groups
distillation_loss = MultiGrpLoss
distillation_loss_args = dict(
    grp_idxs = grp_idxs,
    loss_fns = (
        (PixelWiseKLDiv, dict(log_target=False)),
        (PixelWiseKLDiv, dict(log_target=False)),
        (PixelWiseKLDiv, dict(log_target=False)),
        (PixelWiseKLDiv, dict(log_target=False)),
    ),
    split_targets=True,
)


# Combining both main loss and self distillation function into one master function
loss_args = dict(n_layers=6, target='out', segmentation_loss = (segmentation_loss, segmentation_loss_args), distillation_loss = (distillation_loss, distillation_loss_args), temperature = 3, weights_segmentation_loss = weights_segmentation_loss, weights_distillation_loss = weights_distillation_loss, return_just_loss = True,)
loss_fn = CompositeLoss(**loss_args, encoder_keys_for_self_distillation = [f'encoder'])
# [CHECK THIS]

# Setting up hyperparameteres for the MAGIC framwork
modality_names: list[str] = [key[0] for key in training_sets] # i.e. ['VR', 'CCTA', 'simCT]
agnostic_name: str = 'Mask'
grp_seperator = GroupSeperate(grp_idxs)
num_classes_set: list[int] = 1 + 9 + 8 + 2 + 4 # Total number of classes involved, i.e. WH, Chambers/great vessels, Coronary arteries/valves, nodes, AND backgrounds
split_level = 4 # How deep in the network to split the model for each output group
# -> Configured to be a 6 layer nnU-Net base, i=4 puts the split right after the bottle neck
out_groups = [2, 10, 9, 3] # Number classes + bkg for each group to guide the output layers of the decoders
# 1 WH, 9 Chambers + Great Vessels, 8 coronary arteries + valves, 2 nodes, 4 "groups"

# Needed parameters for the backbone blocks
model_hyperparams = dict(
    scale_factors=[1, 2, 4, 8, 16],
    spatial_dims=3,
    filters = [32, 64, 128, 256, 512, 512],
    in_channels = 1,
    out_channels = num_classes_set,
    strides = [(1, 1, 1)] + [(2, 2, 2) for _ in range(5)],
    kernel_size = [(3, 3, 3) for _ in range(6)],
) 

# Initialize the MAGIC 
magic = MAGIC_framework(
    roi_size=roi_size,
    modality_names=modality_names,
    agnostic_name='Mask',
    encoder_module=(SplitDynEncoder_wSD, model_hyperparams | {'attention_block': False}),
    decoder_module=(SplitDynDecoder_wSD, model_hyperparams | {'split_level': split_level, 'out_groups': out_groups}),
    out_caller=out_caller,
    optimizer_class=(AdamW, dict(lr=1e-3, weight_decay=1e-4, amsgrad=True)),
    learning_rate_class=(torch.optim.lr_scheduler.LambdaLR, dict(lr_lambda=lr_lambda(1000, gamma=0.9))),
    loss_function=loss_fn, # The loss functions -> Not necessarily needed, used to help with organization
).to(device)

# Esentially, MAGIC is configured into the following elements:
# Blocks: VR Encoder [E1], CCTA Encoder [E2], Sim-CT Encoder [E3], Shared Decoder [D]
# Input: x; Output: y
# Models:
#   VR: y = D(E1(x))
#   CCTA: y = D(E2(x))
#   SimCT: y = D(E3(x))

# The optimizer connects all the blocks such that regardless of what modality is run, it can keep track of the moment along the way

# ----------------------------------------------------------------
# Output setup
# ----------------------------------------------------------------
# Defining what to do after the prediction: i.e. tracking history / validation
loss_weights = {}
loss_history = {modality: [] for modality in modality_names} | {'weights': loss_weights}
label_OneHot_fns = {i: AsDiscrete(to_onehot=len(grp)) for i, grp in enumerate(grp_idxs)}

dice_fn = DiceMetric(include_background = True, reduction = 'none', get_not_nans = False)
hd95_fn = HausdorffDistanceMetric(include_background = True, percentile=95, reduction = 'none')

metric_fns: dict[str, CumulativeIterationMetric] = {
    'dice': dice_fn,
    'hd95': hd95_fn
}
comparisons: dict[str, Callable[[Any, Any], bool]] = {
    'dice': operator.gt,
    'hd95': operator.lt
}
replace_nan: dict[str, Union[None, float]] = {
    'dice': None,
    'hd95': 256
}

# When defining the `best_val_history`, the `best_val` (the value that needs to be beet) either needs to be 0 if the metric is increasing, or 1 if it is decreassing
# The operators do not have a direct type to check. Instead, I pass "comparison[met](1, 0)". 
# If it is true, then 1 is greater than 0, and the operator is `greater than`.
# Under that situation, you are checking for values that are larger than it. So the starting point should be 0.
# Otherwise, the operator is `less then` and you are looking for smaller. So the starting point should be large (1e9)
# best_val_history = {modality: {met: {'best_val': 0 if comparisons[met](1, 0) else 1e9, 'best_epoch': 0} for met in metric_fns.keys()} for modality in forest.config['modality_names']}
best_val_history = {met: {'best_val': -1 if comparisons[met](1, 0) else 1e9, 'best_epoch': 0} for met in metric_fns.keys()}
val_history = []

# ----------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------
num_epochs = 500
training_start = datetime.datetime.now()
for epoch in range(num_epochs):
    
    cout(f'Current epoch: {epoch}', color=color_options.PINK)
    epoch_start = datetime.datetime.now()

    # TRAINING STEP
    iter_loss_hist = {m: [] for m in magic.config['modality_names']}
    # for modality, training_dataloader in training_sets:
    with alive_bar(len(training_dataloader)) as bar:
        for modality, ds in training_dataloader:
            image = ds['image'].to(device)
            targets: list[torch.Tensor] = [ds[f'grp_{_l}'].to(device) for _l in range(len(grp_idxs))]

            output = magic.tricks[modality](image, for_training=True)
            output = format_training_output_nnUNetwSD(output)

            magic.optimizer.zero_grad()

            loss: torch.Tensor = magic.loss_function(output, targets)
            iter_loss_hist[modality].append(loss.item())
            
            loss.backward()
            magic.optimizer.step()

            bar()

    magic.optimizer_LRS.step()

    # Training clean up
    for modality in magic.config['modality_names']:
        cout(f'For {modality}...')
        # Track the loss history and report
        loss_history[modality].append(np.mean(iter_loss_hist[modality]))            
        cout(f'    loss: {loss_history[modality][-1]:0.3e}')
    
    cout(f'\nTime for training: {datetime.datetime.now() - epoch_start}')
    cout()


    # --------------------------------
    # Validation step
    # --------------------------------
    validation_start = datetime.datetime.now()

    with torch.no_grad():
        
        # Reset the metrics
        for k in metric_fns.keys(): metric_fns[k].reset()

        # Go through each modality
        with alive_bar(len(validation_dataloader)) as bar:
            for modality, ds in validation_dataloader:
                image = ds['image'].to(device)
                master_labels: list[torch.Tensor] = [label_OneHot_fns[i](ds[f'grp_{i}'][0])[1:].unsqueeze(0).to(device) for i in range(len(grp_idxs))]
                
                # Get a full prediction on the input
                level_prediction = sliding_window_inference(inputs = image, roi_size=magic.config['roi_size'], sw_batch_size=batch_size, predictor=magic.tricks[modality], overlap=0.5)
                grp_predictions = grp_seperator(level_prediction)

                predictions = []
                for ii in range(len(grp_idxs)):
                    _out = label_OneHot_fns[ii](torch.argmax(grp_predictions[ii], dim=1))[1:].unsqueeze(0)
                    predictions.append(_out)

                targets = torch.concatenate(master_labels, dim=1)
                predictions = torch.concatenate(predictions, dim=1)

                # Calculate the metrics
                for k in metric_fns.keys(): metric_fns[k](predictions, targets)

                # End
                bar()

        metric_out: dict[str, torch.Tensor] = {k: metric_fns[k].aggregate().cpu().mean(0) for k in metric_fns.keys()}
        for met_name, met_out in metric_out.items():
            cout(f'  Looking at {met_name} results...')
            comp = comparisons[met_name]
            current_val = met_out.mean().item()
            val_history.append(current_val)

            old_best_val = best_val_history[met_name]['best_val']
            old_best_epoch = best_val_history[met_name]['best_epoch']

            if comp(current_val, old_best_val):
                cout(f'    NEW best validation score found for {met_name}! {current_val:0.4f} > {old_best_val:0.4f} @ {old_best_epoch}', color=color_options.GREEN, bold=True, underline=True)
                best_val_history[met_name]['best_val'] = current_val
                best_val_history[met_name]['best_epoch'] = epoch
                magic.save(dst=os.path.join(save_dir, f'Best_Val_{met_name}'), save_weights_only=True, info={'current_epoch': epoch}, save_for_training=False)
            else:
                cout(f'    No new best validation score found for {met_name} :( {current_val:0.4f} < {old_best_val:0.4f} @ {old_best_epoch}', color=color_options.RED)

        cout()

        magic.save(dst=os.path.join(save_dir, f'Last_Model'), save_weights_only=True, info={'current_epoch': epoch}, save_for_training=False)
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(dict(best_val_history=best_val_history, loss_history=loss_history, val_history=val_history), f, indent=2)

        # going through the metrics
        end = datetime.datetime.now()
        cout(f'\nTime for validation: {end - validation_start}')
        cout(f'Total time for epoch: {end - epoch_start}')
        cout()
        cout('*'*64)
        cout()

cout(f'Training done')
cout(f'Time for all {num_epochs} epochs: {datetime.datetime.now() - training_start}')