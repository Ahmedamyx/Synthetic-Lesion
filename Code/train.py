import os
import nibabel as nib
from glob import glob
import torch
from tqdm import tqdm
from datetime import datetime
import random
from monai.data import Dataset, DataLoader, pad_list_data_collate
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.networks.nets import AttentionUnet

from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    AsDiscreted,
    Spacingd,
    EnsureChannelFirstd,
    LoadImaged,
    Lambdad,
    ScaleIntensityd,
    RandGaussianNoised,
    Affined,
    RandCropByLabelClassesd,
    ConcatItemsd,
    RandShiftIntensityd,
    RandScaleIntensityd,
)

import numpy as np
import matplotlib.pyplot as plt

import csv, json
import torch.nn.functional as F 
from monai.losses import DiceLoss
from torch.utils.tensorboard import SummaryWriter

import wandb 

from torch.utils.data import WeightedRandomSampler
from monai.transforms import RandAffined, EnsureTyped 

# --------- Hard-coded Constants ---------

working_dir = '.'
dataset_dir = r'C:\Users\ahmed\Downloads\PDS Ahmed 1\PDS Ahmed\ISLES-2022'
model_checkpoint_path = r'C:\Users\ahmed\Downloads\PDS Ahmed 1\PDS Ahmed\checkpoints'

# Training / Validation Splits & Parameters
IS_SHUFFLE = True
SHUFFLE_SEED_BASE = 32  

ISLES_TRAINING_SIZE = 220
BERN_TRAINING_SIZE  = 250
CHUV_TRAINING_SIZE  = 75

TRAINING_REPEAT   = 2
VALIDATION_REPEAT = 2

TRAINING_SPATIAL_SIZE   = (96, 96, 16)
VALIDATION_SPATIAL_SIZE = (96, 96, 16)

TRAINING_NEG_POS_RATIO   = [1, 2]
VALIDATION_NEG_POS_RATIO = [1, 100]

BATCH_SIZE = 2

# Model & Optimization Hyperparameters
MAX_EPOCHS      = 200
COSINE_LR_T_MAX = MAX_EPOCHS
LEARNING_RATE   = 0.01
VAL_INTERVAL    = 1  
LAMBDA_DICE  = 0.3
LAMBDA_FOCAL = 0.7
LAMBDA_CE    = 0.3
DICE_METRIC_IGNORE_EMPTY = True
DISCRETE_THRESH = 0.5

IS_LOAD_MODEL = False
LOAD_MODEL_PATH = ''
IS_FREEZE = False

VOXEL_DIMENSIONS = (1.0, 1.0, 2.0)

MODEL_TYPE = 'AttentionUnet'

MODEL_PARAMS = dict(
    spatial_dims=3,
    in_channels=2,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    dropout=0.1
)


# --------- Utility Functions ---------
'''
def bern_filter_valid_data(data_path):
    """
    Filters out invalid cases in the Bern dataset.
    """
    adc_dir = os.path.join(data_path, 'ADC_Path')
    dwi_dir = os.path.join(data_path, 'DWI_Path')
    seg_dir = os.path.join(data_path, 'Segmentation_Path')

    images_ADC_paths = []
    images_DWI_paths = []
    segs_paths = []

    adc_files = sorted(os.listdir(adc_dir))
    dwi_files = sorted(os.listdir(dwi_dir))
    seg_files = sorted(os.listdir(seg_dir))

    for adc_file, dwi_file, seg_file in zip(adc_files, dwi_files, seg_files):
        adc_path = os.path.join(adc_dir, adc_file)
        dwi_path = os.path.join(dwi_dir, dwi_file)
        seg_path = os.path.join(seg_dir, seg_file)

        if not (os.path.exists(adc_path) and os.path.exists(dwi_path) and os.path.exists(seg_path)):
            continue

        try:
            adc_image = nib.load(adc_path).get_fdata()
            dwi_image = nib.load(dwi_path).get_fdata()
            seg_image = nib.load(seg_path).get_fdata()
        except nib.filebasedimages.ImageFileError:
            continue

        if adc_image.shape != dwi_image.shape or adc_image.shape != seg_image.shape:
            continue

        images_ADC_paths.append(adc_path)
        images_DWI_paths.append(dwi_path)
        segs_paths.append(seg_path)

    print(f'There are {len(images_ADC_paths)} valid cases available (Bern)')
    return images_ADC_paths, images_DWI_paths, segs_paths
'''

def isles_filter_valid_data(data_path, derivatives_path=r'\Users\ahmed\Downloads\PDS Ahmed 1\PDS Ahmed\ISLES-2022\derivatives'):
    """
    ISLES dataset - modified for your folder structure.
    
    data_path: Path to ISLES-2022 folder containing sub-strokecaseXXXX folders
    derivatives_path: Path to derivatives folder containing segmentation masks
    
    Returns: images_ADC_paths, images_DWI_paths, segs_paths
    """
    images_ADC_paths = []
    images_DWI_paths = []
    segs_paths = []
    
    # Find all subject folders (sub-strokecaseXXXX)
    subject_folders = []
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path) and item.startswith("sub-strokecase"):
            subject_folders.append(item)
    
    # Sort subject folders by case number
    subject_folders = sorted(subject_folders, key=lambda x: int(x.split("sub-strokecase")[1]))
    
    print(f"Found {len(subject_folders)} subject folders")
    
    # Process each subject
    for subject in subject_folders:
        subject_id = subject  # e.g., "sub-strokecase0001"
        
        # Build paths for ADC and DWI files
        adc_path = os.path.join(data_path, subject, "ses-0001", "dwi", f"{subject}_ses-0001_adc.nii.gz")
        dwi_path = os.path.join(data_path, subject, "ses-0001", "dwi", f"{subject}_ses-0001_dwi.nii.gz")
        
        # Build path for segmentation file in derivatives folder
        seg_path = os.path.join(derivatives_path, subject, "ses-0001", f"{subject}_ses-0001_msk.nii.gz")
        
        # Check if ADC file exists
        if os.path.exists(adc_path):
            images_ADC_paths.append(adc_path)
        else:
            print(f"Warning: ADC file not found for {subject}: {adc_path}")
        
        # Check if DWI file exists
        if os.path.exists(dwi_path):
            images_DWI_paths.append(dwi_path)
        else:
            print(f"Warning: DWI file not found for {subject}: {dwi_path}")
        
        # Check if segmentation file exists
        if os.path.exists(seg_path):
            segs_paths.append(seg_path)
        else:
            # Try alternative naming (without .gz extension)
            seg_path_alt = os.path.join(derivatives_path, subject, "ses-0001", f"{subject}_ses-0001_msk.nii")
            if os.path.exists(seg_path_alt):
                segs_paths.append(seg_path_alt)
            else:
                print(f"Warning: Segmentation file not found for {subject}: {seg_path}")
    
    # Verify all lists have the same length
    if len(images_ADC_paths) != len(images_DWI_paths) or len(images_ADC_paths) != len(segs_paths):
        print(f"Warning: Mismatch in file counts!")
        print(f"  ADC files: {len(images_ADC_paths)}")
        print(f"  DWI files: {len(images_DWI_paths)}")
        print(f"  Segmentation files: {len(segs_paths)}")
        
        # Find which subjects have complete data
        complete_subjects = []
        for subject in subject_folders:
            adc_exists = os.path.exists(os.path.join(data_path, subject, "ses-0001", "dwi", f"{subject}_ses-0001_adc.nii.gz"))
            dwi_exists = os.path.exists(os.path.join(data_path, subject, "ses-0001", "dwi", f"{subject}_ses-0001_dwi.nii.gz"))
            seg_exists = os.path.exists(os.path.join(derivatives_path, subject, "ses-0001", f"{subject}_ses-0001_msk.nii.gz")) or \
                        os.path.exists(os.path.join(derivatives_path, subject, "ses-0001", f"{subject}_ses-0001_msk.nii"))
            
            if adc_exists and dwi_exists and seg_exists:
                complete_subjects.append(subject)
        
        print(f"Subjects with complete data: {len(complete_subjects)}")
        print(f"Complete subjects: {complete_subjects}")
        
        # Rebuild lists with only complete subjects
        images_ADC_paths = []
        images_DWI_paths = []
        segs_paths = []
        
        for subject in complete_subjects:
            adc_path = os.path.join(data_path, subject, "ses-0001", "dwi", f"{subject}_ses-0001_adc.nii.gz")
            dwi_path = os.path.join(data_path, subject, "ses-0001", "dwi", f"{subject}_ses-0001_dwi.nii.gz")
            
            # Try .gz first, then .nii
            seg_path = os.path.join(derivatives_path, subject, "ses-0001", f"{subject}_ses-0001_msk.nii.gz")
            if not os.path.exists(seg_path):
                seg_path = os.path.join(derivatives_path, subject, "ses-0001", f"{subject}_ses-0001_msk.nii")
            
            images_ADC_paths.append(adc_path)
            images_DWI_paths.append(dwi_path)
            segs_paths.append(seg_path)
    
    print(f"\nFinal counts:")
    print(f"  ADC files: {len(images_ADC_paths)}")
    print(f"  DWI files: {len(images_DWI_paths)}")
    print(f"  Segmentation files: {len(segs_paths)}")
    
    return images_ADC_paths, images_DWI_paths, segs_paths

'''
def CHUV_filter_valid_data(data_path):
    """
    Filters out invalid cases in the CHUV dataset.
    """
    adc_dir = os.path.join(data_path, 'ADC')
    dwi_dir = os.path.join(data_path, 'Trace')
    seg_dir = os.path.join(data_path, 'Segmentation')

    images_ADC_paths = []
    images_DWI_paths = []
    segs_paths = []

    adc_files = sorted(os.listdir(adc_dir))
    dwi_files = sorted(os.listdir(dwi_dir))
    seg_files = sorted(os.listdir(seg_dir))

    for adc_file, dwi_file, seg_file in zip(adc_files, dwi_files, seg_files):
        adc_path = os.path.join(adc_dir, adc_file)
        dwi_path = os.path.join(dwi_dir, dwi_file)
        seg_path = os.path.join(seg_dir, seg_file)

        if not (os.path.exists(adc_path) and os.path.exists(dwi_path) and os.path.exists(seg_path)):
            continue

        try:
            adc_image = nib.load(adc_path).get_fdata()
            dwi_image = nib.load(dwi_path).get_fdata()
            seg_image = nib.load(seg_path).get_fdata()
        except nib.filebasedimages.ImageFileError:
            continue

        if adc_image.shape != dwi_image.shape or adc_image.shape != seg_image.shape:
            continue

        images_ADC_paths.append(adc_path)
        images_DWI_paths.append(dwi_path)
        segs_paths.append(seg_path)

    print(f'There are {len(images_ADC_paths)} valid cases available (CHUV)')
    return images_ADC_paths, images_DWI_paths, segs_paths
'''

def evaluate_loader(model, loader, dice_metric, dice_metric_all, post_trans, device, val_precision, val_recall, val_step_all, bce_loss_fn, dice_loss_fn):
    """
    Evaluate the model for the three sites.
    """
    #val_step = 0

    ce_sum = 0.0
    dloss_sum = 0.0

    for val_batch_data in tqdm(loader):
      # val_step += 1
        val_step_all += 1

        val_inputs = val_batch_data["adc_dwi"].to(device)
        val_labels = val_batch_data["seg"].to(device).float()

        with torch.no_grad():
            logits = model(val_inputs)
            ce_sum += bce_loss_fn(logits, val_labels).item()
            dloss_sum += dice_loss_fn(logits, val_labels).item()

            preds = post_trans(logits)

            dice_metric(y_pred=preds, y=val_labels)
            dice_metric_all(y_pred=preds, y=val_labels)

            prec, rec = calculate_precision_recall(val_labels, preds)
            val_precision += prec
            val_recall += rec

    val_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return val_dice, val_precision, val_recall, val_step_all, ce_sum, dloss_sum 


def calculate_precision_recall(y_true, y_pred):
    """
    Calculate Precision and Recall given two binary tensors.
    """
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)

    TP = ((y_true_flat == 1) & (y_pred_flat == 1)).sum().item()
    FP = ((y_true_flat == 0) & (y_pred_flat == 1)).sum().item()
    FN = ((y_true_flat == 1) & (y_pred_flat == 0)).sum().item()

    epsilon = 1e-7
    precision = TP / (TP + FP + epsilon)
    recall    = TP / (TP + FN + epsilon)
    return precision, recall

# ---------- Utility Lambdas ----------
def lambda_rescale_adc(image):
    image[image < 0] = 0
    min_input = image.min()
    max_input = image.max()
    maxv, minv = 4000, 0
    scale = (maxv - minv) / (max_input - min_input)
    offset = minv - min_input * scale
    return image * scale + offset

def lambda_rescale_dwi(image):
    image[image < 0] = 0
    min_input = image.min()
    max_input = image.max()
    maxv, minv = 2000, 0
    scale = (maxv - minv) / (max_input - min_input)
    offset = minv - min_input * scale
    return image * scale + offset

def lambda_clamp_adc(image):
    image[image < 10] = 0
    image[image > 3000] = 3000
    return image

def lambda_clamp_dwi(image):
    image[image < 5] = 0
    image[image > 1800] = 1800
    return image
def create_dataloader(random_seed):
    """
    Create DataLoaders for train / validation splits of Bern, ISLES, and CHUV sets.
    """
    
    def _has_foreground(seg_path): 
        try: 
            m = nib.load(seg_path).get_fdata()
            return bool((m>0).any())
        except Exception: 
            return False 

    # ---------- Collect Data Paths ----------
    # chuv_ADC, chuv_DWI, chuv_seg = CHUV_filter_valid_data(os.path.join(dataset_dir, 'Train', 'CHUV'))
    isles_ADC, isles_DWI, isles_seg = isles_filter_valid_data(os.path.join(dataset_dir))
    # bern_ADC, bern_DWI, bern_seg = bern_filter_valid_data(os.path.join(dataset_dir, 'Train', 'Bern'))

    # ---------- Shuffle Datasets (optional, necessary for multiple trials) ----------
    if IS_SHUFFLE:
        random.seed(random_seed)
        random.shuffle(isles_ADC)
        random.seed(random_seed)
        random.shuffle(isles_DWI)
        random.seed(random_seed)
        random.shuffle(isles_seg)
    '''
        random.seed(random_seed)
        random.shuffle(bern_ADC)
        random.seed(random_seed)
        random.shuffle(bern_DWI)
        random.seed(random_seed)
        random.shuffle(bern_seg)

        random.seed(random_seed)
        random.shuffle(chuv_ADC)
        random.seed(random_seed)
        random.shuffle(chuv_DWI)
        random.seed(random_seed)
        random.shuffle(chuv_seg)
    '''
    # ---------- Split Train / Validation ----------
    # ISLES
    isles_ADC_train = isles_ADC[:ISLES_TRAINING_SIZE]
    isles_DWI_train = isles_DWI[:ISLES_TRAINING_SIZE]
    isles_seg_train = isles_seg[:ISLES_TRAINING_SIZE]

    isles_ADC_val = isles_ADC[ISLES_TRAINING_SIZE:]
    isles_DWI_val = isles_DWI[ISLES_TRAINING_SIZE:]
    isles_seg_val = isles_seg[ISLES_TRAINING_SIZE:]
    '''
    # Bern
    bern_ADC_train = bern_ADC[:BERN_TRAINING_SIZE]
    bern_DWI_train = bern_DWI[:BERN_TRAINING_SIZE]
    bern_seg_train = bern_seg[:BERN_TRAINING_SIZE]

    bern_ADC_val = bern_ADC[BERN_TRAINING_SIZE:]
    bern_DWI_val = bern_DWI[BERN_TRAINING_SIZE:]
    bern_seg_val = bern_seg[BERN_TRAINING_SIZE:]

    # CHUV
    chuv_ADC_train = chuv_ADC[:CHUV_TRAINING_SIZE]
    chuv_DWI_train = chuv_DWI[:CHUV_TRAINING_SIZE]
    chuv_seg_train = chuv_seg[:CHUV_TRAINING_SIZE]

    chuv_ADC_val = chuv_ADC[CHUV_TRAINING_SIZE:]
    chuv_DWI_val = chuv_DWI[CHUV_TRAINING_SIZE:]
    chuv_seg_val = chuv_seg[CHUV_TRAINING_SIZE:]

    # Combine
    images_ADC_train = isles_ADC_train + bern_ADC_train + chuv_ADC_train
    images_DWI_train = isles_DWI_train + bern_DWI_train + chuv_DWI_train
    segs_train       = isles_seg_train + bern_seg_train + chuv_seg_train

    # Shuffle the combined train data once more
    random.seed(random_seed)
    random.shuffle(images_ADC_train)
    random.seed(random_seed)
    random.shuffle(images_DWI_train)
    random.seed(random_seed)
    random.shuffle(segs_train)
    '''
    train_files = [
        {"adc": a, "dwi": b, "seg": c} 
        for a, b, c in zip(isles_ADC_train, isles_DWI_train , isles_seg_train)
    ] * TRAINING_REPEAT
    '''
    # Repeat
    train_files = [
        {"adc": a, "dwi": b, "seg": c} 
        for a, b, c in zip(images_ADC_train, images_DWI_train, segs_train)
    ] * TRAINING_REPEAT
    '''
    # Validation sets
    isles_val_files = [
        {"adc": a, "dwi": b, "seg": c} 
        for a, b, c in zip(isles_ADC_val, isles_DWI_val, isles_seg_val)
    ] * VALIDATION_REPEAT
    '''
    bern_val_files = [
        {"adc": a, "dwi": b, "seg": c} 
        for a, b, c in zip(bern_ADC_val, bern_DWI_val, bern_seg_val)
    ] * VALIDATION_REPEAT

    chuv_val_files = [
        {"adc": a, "dwi": b, "seg": c}
        for a, b, c in zip(chuv_ADC_val, chuv_DWI_val, chuv_seg_val)
    ] * VALIDATION_REPEAT
    '''
    # ---------- Transforms ----------
    train_transforms = Compose([
        LoadImaged(keys=["adc", "dwi", "seg"], image_only=False),
        EnsureChannelFirstd(keys=["adc", "dwi", "seg"]),
        Spacingd(
            keys=["adc", "dwi", "seg"],
            pixdim=VOXEL_DIMENSIONS,
            mode=("bilinear", "bilinear", "nearest"),
        ),
        Lambdad(keys=["adc"], func=lambda_rescale_adc),
        Lambdad(keys=["dwi"], func=lambda_rescale_dwi),
        RandAffined(keys=["adc", "dwi", "seg"],prob=0.3, rotate_range=(0.1, 0.1, 0.1), shear_range=(0.05, 0.05, 0.05), scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "bilinear", "nearest"), padding_mode='border'),
        RandGaussianNoised(keys=["adc", "dwi"], prob=0.5, mean=0., std=1),
        Lambdad(keys=["adc"], func=lambda_clamp_adc),
        Lambdad(keys=["dwi"], func=lambda_clamp_dwi),
        ScaleIntensityd(keys=["adc", "dwi"]),
        AsDiscreted(keys=["seg"], threshold=0.5),
        RandCropByLabelClassesd(
            keys=["adc", "dwi", "seg"], 
            label_key="seg", 
            spatial_size=TRAINING_SPATIAL_SIZE,
            ratios=TRAINING_NEG_POS_RATIO,
            num_classes=2, 
            num_samples=8
        ),
        EnsureTyped(keys=["adc", "dwi", "seg"]),
        ConcatItemsd(keys=["adc", "dwi"], name="adc_dwi", dim=0),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["adc", "dwi", "seg"], image_only=False),
        EnsureChannelFirstd(keys=["adc", "dwi", "seg"]),
        Spacingd(
            keys=["adc", "dwi", "seg"],
            pixdim=VOXEL_DIMENSIONS,
            mode=("bilinear", "bilinear", "nearest"),
        ),
        Lambdad(keys=["adc"], func=lambda_rescale_adc),
        Lambdad(keys=["dwi"], func=lambda_rescale_dwi),
        Lambdad(keys=["adc"], func=lambda_clamp_adc),
        Lambdad(keys=["dwi"], func=lambda_clamp_dwi),
        ScaleIntensityd(keys=["adc", "dwi"]),
        AsDiscreted(keys=["seg"], threshold=0.5),
        RandCropByLabelClassesd(
            keys=["adc", "dwi", "seg"], 
            label_key="seg", 
            spatial_size=VALIDATION_SPATIAL_SIZE,
            ratios=VALIDATION_NEG_POS_RATIO, 
            num_classes=2, 
            num_samples=8
        ),

        ConcatItemsd(keys=["adc", "dwi"], name="adc_dwi", dim=0),
    ])

    # ---------- Create Dataloaders ----------
    train_ds      = Dataset(data=train_files,       transform=train_transforms)
    isles_val_ds  = Dataset(data=isles_val_files,   transform=val_transforms)
    # bern_val_ds   = Dataset(data=bern_val_files,    transform=val_transforms)
    # chuv_val_ds   = Dataset(data=chuv_val_files,    transform=val_transforms)

    POS_VOL_WEIGHT = 3.0
    weights = [POS_VOL_WEIGHT if _has_foreground(d["seg"]) else 1.0 for d in train_files]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader      = DataLoader(train_ds,     batch_size=BATCH_SIZE, sampler=sampler, collate_fn=pad_list_data_collate, num_workers=4, pin_memory=True)
    isles_val_loader  = DataLoader(isles_val_ds, batch_size=1,          collate_fn=pad_list_data_collate, num_workers=2,  pin_memory=True)
    # bern_val_loader   = DataLoader(bern_val_ds,  batch_size=1,          collate_fn=pad_list_data_collate, num_workers=8,  pin_memory=True)
    # chuv_val_loader   = DataLoader(chuv_val_ds,  batch_size=1,          collate_fn=pad_list_data_collate, num_workers=8,  pin_memory=True)

    return train_loader, isles_val_loader
    # return train_loader, isles_val_loader, bern_val_loader, chuv_val_loader



def create_model():
    """
    Creates and returns a hard-coded AttentionUnet model (or other). 
    Also loads weights if IS_LOAD_MODEL is True.
    """
    # Example: Hard-coded model
    model = AttentionUnet(**MODEL_PARAMS)

    if IS_LOAD_MODEL:
        print(f'Loading model from {LOAD_MODEL_PATH} for finetuning...')
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        if IS_FREEZE:
            # Example: freeze a part of the model if needed
            for name, param in model.named_parameters():
                # Adjust the condition to match the layers you want to freeze
                if 'model.1.submodule.1.submodule.1' in name:
                    param.requires_grad = False
    else:
        print('Training model from scratch...')

    return model


def objective(train_id: int):
    """
    Full training loop for a single 'trial' or run.
    """
    # Model & Device
    model = create_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # W&B run 
    run = wandb.init(
        project="stroke-lesion-segmentation1", 
        name=f"{time_string}-trials-{train_id}",
        group=time_string,
        reinit=True,
        config=dict(
            MAX_EPOCHS=MAX_EPOCHS, BATCH_SIZE=BATCH_SIZE, LR=LEARNING_RATE, COSINE_LR_T_MAX=COSINE_LR_T_MAX, 
            VAL_INTERVAL=VAL_INTERVAL, LAMBDA_DICE=LAMBDA_DICE, LAMBDA_CE=LAMBDA_CE, 
            SPATIAL_SIZE_TRAIN=TRAINING_SPATIAL_SIZE, SPATIAL_SIZE_VAL=VALIDATION_SPATIAL_SIZE,
            VOXEL_DIMENSIONS=VOXEL_DIMENSIONS, MODEL="AttentionUnet", MODEL_PARAMS=MODEL_PARAMS
        )
    )
    wandb.watch(model, log="gradients", log_freq=100)

    # Create Dataloaders (shuffling depends on seed = SHUFFLE_SEED_BASE + train_id)
    random_seed = SHUFFLE_SEED_BASE + train_id
    train_loader, isles_val_loader = create_dataloader(random_seed)
    # train_loader, isles_val_loader, bern_val_loader, chuv_val_loader = create_dataloader(random_seed)

    loss_function = DiceCELoss(sigmoid=True, to_onehot_y=False, include_background=False, jaccard=False, lambda_dice=LAMBDA_DICE, lambda_ce=LAMBDA_CE)

    # For separate logging 
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss(sigmoid=True, include_background=False, reduction="mean") 

    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=DICE_METRIC_IGNORE_EMPTY)
    dice_metric_all = DiceMetric(include_background=False, reduction="mean", ignore_empty=DICE_METRIC_IGNORE_EMPTY)

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=COSINE_LR_T_MAX)

    # Training state
    best_val_metric = -1
    best_val_epoch  = -1

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=DISCRETE_THRESH)])

    # === logging setup (per trail dirs already created in main) ===
    trial_dir = os.path.join(model_checkpoint_path, time_string, f"trial_{train_id}")
    csv_path = os.path.join(trial_dir, "history.csv")
    tb_writer = SummaryWriter(log_dir=os.path.join(trial_dir,"tb"))

    # init CSV headerp
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f: 
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_ce", "train_dice_loss", "train_dice", "train_precision", "train_recall", 
                        "val_dice", "val_precision", "val_recall", "val_ce", "val_dice_loss", "lr"])

    for epoch in range(MAX_EPOCHS):
        print("-" * 100)
        print(f"epoch {epoch + 1}/{MAX_EPOCHS}")

        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_dice_loss = 0.0
        epoch_precision = 0.0
        epoch_recall = 0.0
        step = 0

        scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda")) 

        # --------------------- TRAIN ---------------------
        for batch_data in tqdm(train_loader):
            step += 1
            inputs = batch_data["adc_dwi"].to(device)
            labels = batch_data["seg"].to(device).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(inputs)
                loss = loss_function(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # separate CE / Dice loss for logging only 
            with torch.no_grad(): 
                ce_val = bce_loss_fn(logits, labels).item()
                dloss_val = dice_loss_fn(logits, labels).item()
                epoch_ce += ce_val
                epoch_dice_loss += dloss_val

                preds = post_trans(logits)
                dice_metric(y_pred=preds, y=labels)
                prec, rec = calculate_precision_recall(labels, preds)
                epoch_precision += prec
                epoch_recall += rec 

        # averages 
        epoch_loss /= step
        epoch_ce /= step 
        epoch_dice_loss /= step 
        epoch_precision /= step
        epoch_recall /= step
        epoch_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        wandb.log({
            "train/loss": epoch_loss, 
            "train/ce": epoch_ce, 
            "train/dice_loss": epoch_dice_loss,
            "train/dice": epoch_dice, 
            "train/precision": epoch_precision, 
            "train/recall": epoch_recall, 
            "lr": scheduler.get_last_lr()[0],
        }, step=epoch+1)


        print(f"epoch {epoch + 1} train: loss={epoch_loss:.4f} | dice={epoch_dice:.4f} | CE={epoch_ce:.4f} | dLoss={epoch_dice_loss:.4f} | P={epoch_precision:.4f} | R={epoch_recall:.4f}")

        # Tensorboard logging 
        tb_writer.add_scalar("train/loss", epoch_loss, epoch+1)
        tb_writer.add_scalar("train/ce_loss", epoch_ce, epoch+1)
        tb_writer.add_scalar("train/dice_loss", epoch_dice_loss, epoch+1)
        tb_writer.add_scalar("train/dice", epoch_dice, epoch+1)
        tb_writer.add_scalar("train/precision", epoch_precision, epoch+1)
        tb_writer.add_scalar("train/recall", epoch_recall, epoch+1)
        tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch+1)

        # --------------------- VALIDATION ---------------------
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                dice_metric.reset()
                dice_metric_all.reset()

                val_precision = 0.0
                val_recall = 0.0
                val_step_all = 0

                # Evaluate on ISLES
                val_dice_isles, val_precision, val_recall, val_step_all, val_ce_sum, val_dloss_sum = evaluate_loader(
                    model=model,
                    loader=isles_val_loader,
                    dice_metric=dice_metric,
                    dice_metric_all=dice_metric_all,
                    post_trans=post_trans,
                    device=device,
                    val_precision=val_precision,
                    val_recall=val_recall,
                    val_step_all=val_step_all,
                    bce_loss_fn=bce_loss_fn, 
                    dice_loss_fn=dice_loss_fn
                )
            
                '''
                # Evaluate on Bern
                val_dice_bern, val_precision, val_recall, val_step_all = evaluate_loader(
                    model=model,
                    loader=bern_val_loader,
                    dice_metric=dice_metric,
                    dice_metric_all=dice_metric_all,
                    post_trans=post_trans,
                    device=device,
                    val_precision=val_precision,
                    val_recall=val_recall,
                    val_step_all=val_step_all
                )

                # Evaluate on CHUV
                val_dice_chuv, val_precision, val_recall, val_step_all = evaluate_loader(
                    model=model,
                    loader=chuv_val_loader,
                    dice_metric=dice_metric,
                    dice_metric_all=dice_metric_all,
                    post_trans=post_trans,
                    device=device,
                    val_precision=val_precision,
                    val_recall=val_recall,
                    val_step_all=val_step_all
                )
                '''
                # Combined (across all three validation sets)
                val_dice = dice_metric_all.aggregate().item()
                dice_metric_all.reset()

                val_precision /= max(val_step_all, 1)
                val_recall    /= max(val_step_all, 1)
                val_ce = val_ce_sum / max(val_step_all, 1)
                val_dloss = val_dloss_sum / max(val_step_all, 1)

                wandb.log({
                "val/dice": val_dice, 
                "val/ce": val_ce, 
                "val/dice_loss": val_dloss,
                "val/precision": val_precision,
                "val/recall": val_recall,
                }, step=epoch+1)

                print(f"epoch {epoch + 1} val: dice={val_dice:.4f} | CE={val_ce:.4f} | dLoss={val_dloss:.4f} | P={val_precision:.4f} | R={val_recall:.4f}")
              
                print(f"     -> val dice (ISLES): {val_dice_isles:.4f}")
                # print(f"     -> val dice (ISLES): {val_dice_isles:.4f} | (Bern): {val_dice_bern:.4f} | (CHUV): {val_dice_chuv:.4f}\n")

                # CSV row 
                with open(csv_path, "a", newline="") as f: 
                    w = csv.writer(f)
                    w.writerow([epoch+1, epoch_loss, epoch_ce, epoch_dice_loss, epoch_dice, epoch_precision, epoch_recall, 
                                val_dice, val_precision, val_recall, val_ce, val_dloss, scheduler.get_last_lr()[0]])
                    
                # Tensorboard logging 
                tb_writer.add_scalar("val/dice", val_dice, epoch+1)
                tb_writer.add_scalar("val/ce_loss", val_ce, epoch+1)
                tb_writer.add_scalar("val/dice_loss", val_dloss, epoch+1)
                tb_writer.add_scalar("val/precision", val_precision, epoch+1)
                tb_writer.add_scalar("val/recall", val_recall, epoch+1)
                

                # Update Best
                if val_dice > best_val_metric:
                    best_val_metric = val_dice
                    best_val_epoch  = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(trial_dir, f"best_params_e{best_val_epoch}_dice{best_val_metric:.3f}.pth")
                    )   

        # Scheduler step
        scheduler.step()

    with open(os.path.join(trial_dir, "summary.json"), "w") as f: 
        json.dump({
            "best_val_dice": best_val_metric,
            "best_val_epoch": best_val_epoch
        }, f, indent=2)


    print(f"Training finished. Best validation Dice: {best_val_metric:.4f} at epoch {best_val_epoch}.")
    tb_writer.close()


    ckpt_path = os.path.join(trial_dir, f"best_params_e{best_val_epoch}_dice{best_val_metric:.3f}.pth")
    torch.save(model.state_dict(), ckpt_path)

    artifact = wandb.Artifact(
        name=f"attnunet-{time_string}-trial-{train_id}",
        type="model", 
        metadata={"best_val_dice": float(best_val_metric), "epoch": int(best_val_epoch)}
    )

    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact) 

    wandb.run.summary["best_val_dice"] = best_val_metric
    wandb.run.summary["best_val_epoch"] = best_val_epoch
    wandb.finish() 

if __name__ == '__main__':
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M")
    os.makedirs(os.path.join(model_checkpoint_path, time_string), exist_ok=True)

    for i in range(5):
        os.makedirs(os.path.join(model_checkpoint_path, time_string, f"trial_{i}"), exist_ok=True)
        objective(train_id=i)