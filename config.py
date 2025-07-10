import os
from datetime import datetime, timedelta, timezone

utc_offset = timezone(timedelta(hours=5, minutes=30))
current_timestamp = datetime.now(utc_offset).strftime("%Y%m%d_%H%M%S")

workingDirectory = './'
if 'E:' in os.getcwd():
    workingDirectory = 'E:/Projects/PyCharm/FYP/'
elif '/content' in os.getcwd():
    workingDirectory = '/content/'
else:
    workingDirectory = os.getcwd() + '/'


def get_working_directory():
    return workingDirectory


config = {
    "BATCH_SIZE": 16,
    "EPOCHS": 1,
    "DECODER_STAGES": 5,
    "current_timestamp": current_timestamp,
    "plot_sample": False,
    "RATES": [6, 12, 18],
    "LR": 1e-4,
    "includeT1": True,

    "datasetInputTrain": "/kaggle/input/BraTS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/",
    "datasetInputTest": "/kaggle/input/BraTS20/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/",
    "exceptionFolder": "/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355",

    "trainPaths": f"{workingDirectory}BraTS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/",
    "testPaths": f"{workingDirectory}BraTS20/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData",

    "rename355SegSrc": f"{workingDirectory}BraTS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/W39_1998.09.19_Segm.nii",
    "rename355SegTgt": f"{workingDirectory}BraTS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/BraTS20_Training_355_seg.nii",

    "saveNPYimages": f"{workingDirectory}BrainTS/train/input3ch/images/",
    "saveNPYmasks": f"{workingDirectory}BrainTS/train/input3ch/masks/",
    "saveTestNPYimages": f"{workingDirectory}BrainTS/test/input3ch/images/",

    "splitInputFolder": f"{workingDirectory}BrainTS/train/input3ch/",
    "splitOutputFolder": f"{workingDirectory}BrainTS/training/",

    "train_img_dir": f"{workingDirectory}BrainTS/training/train/images/",
    "train_mask_dir": f"{workingDirectory}BrainTS/training/train/masks/",

    "val_img_dir": f"{workingDirectory}BrainTS/training/val/images/",
    "val_mask_dir": f"{workingDirectory}BrainTS/training/val/masks/",

    "test_img_dir": f"{workingDirectory}BrainTS/training/test/images/",
    "test_mask_dir": f"{workingDirectory}BrainTS/training/test/masks/",

    # SET TRAIN AND TEST DATASET PATH
    "loadTrainPath": f"{workingDirectory}BraTS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
    "loadTestPath": f"{workingDirectory}/BraTS20/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData",

    "saveBestModelPath": f"{workingDirectory}BestModels/",

    "threshold": 0.25

}
