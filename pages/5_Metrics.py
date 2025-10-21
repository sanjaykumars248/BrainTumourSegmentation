import os
import json
import pandas as pd
import streamlit as st
import numpy as np
# noinspection PyUnresolvedReferences
from tensorflow.keras import Model
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
import keras
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from config import config

st.set_page_config(layout="wide")

SMOOTH = 1e-6
METRICS = []
thresh = config['threshold']


@keras.saving.register_keras_serializable()
def iou_metric(y_true, y_pred, smooth=1e-6):
    # Flatten the predictions and ground truth tensors
    y_true_f = tf.reshape(y_true, (-1, y_true.shape[-1]))
    y_pred_f = tf.reshape(y_pred, (-1, y_pred.shape[-1]))

    # Convert tensors to float32 for safe computation
    y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = tf.cast(y_pred_f, tf.float32)

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)

    # Calculate IOU for each class
    iou = (intersection + smooth) / (union - intersection + smooth)

    # Return average IOU across all classes
    return tf.reduce_mean(iou)


@keras.saving.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    class_num = y_true.shape[-1]  # Number of classes in the prediction

    # Initialize total Dice coefficient
    total_dice = 0.0

    for i in range(class_num):
        # Flatten the tensors for each class
        y_true_f = tf.keras.layers.Flatten()(y_true[..., i])
        y_pred_f = tf.keras.layers.Flatten()(y_pred[..., i])

        # Cast tensors to float32
        y_true_f = tf.cast(y_true_f, tf.float32)
        y_pred_f = tf.cast(y_pred_f, tf.float32)

        # Compute intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

        # Dice coefficient for current class
        d = (2. * intersection + smooth) / (union + smooth)

        # Accumulate the Dice score
        total_dice += d

    # Return average Dice coefficient across all classes
    return total_dice / class_num


# Dice Coefficients for specific tumor types (necrotic, edema, enhancing)
@keras.saving.register_keras_serializable()
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    y_true_f = K.cast(y_true, 'float32')
    y_pred_f = K.cast(y_pred, 'float32')

    intersection = K.sum(y_true_f[:, :, 1] * y_pred_f[:, :, 1])
    return (2. * intersection + epsilon) / (K.sum(y_true_f[:, :, 1]) + K.sum(y_pred_f[:, :, 1]) + epsilon)


# Dice Coefficient for edema tumor
@keras.saving.register_keras_serializable()
def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    y_true_f = K.cast(y_true, 'float32')
    y_pred_f = K.cast(y_pred, 'float32')

    intersection = K.sum(y_true_f[:, :, 2] * y_pred_f[:, :, 2])
    return (2. * intersection + epsilon) / (K.sum(y_true_f[:, :, 2]) + K.sum(y_pred_f[:, :, 2]) + epsilon)


# Dice Coefficient for enhancing tumor
@keras.saving.register_keras_serializable()
def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    y_true_f = K.cast(y_true, 'float32')
    y_pred_f = K.cast(y_pred, 'float32')

    intersection = K.sum(y_true_f[:, :, 3] * y_pred_f[:, :, 3])
    return (2. * intersection + epsilon) / (K.sum(y_true_f[:, :, 3]) + K.sum(y_pred_f[:, :, 3]) + epsilon)

@keras.saving.register_keras_serializable()
def dice_coef_no_tumour(y_true, y_pred, epsilon=1e-6):
    y_true_f = K.cast(y_true, 'float32')
    y_pred_f = K.cast(y_pred, 'float32')

    intersection = K.sum(y_true_f[:, :, 0] * y_pred_f[:, :, 0])
    return (2. * intersection + epsilon) / (K.sum(y_true_f[:, :, 0]) + K.sum(y_pred_f[:, :, 0]) + epsilon)


# Precision metric
@keras.saving.register_keras_serializable()
def precision(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')

    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_f, 0, 1)))

    return true_positives / (predicted_positives + K.epsilon())


# Sensitivity (Recall) metric
@keras.saving.register_keras_serializable()
def sensitivity(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')

    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))

    return true_positives / (possible_positives + K.epsilon())


# Specificity metric
@keras.saving.register_keras_serializable()
def specificity(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')

    true_negatives = K.sum(K.round(K.clip((1 - y_true_f) * (1 - y_pred_f), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true_f, 0, 1)))

    return true_negatives / (possible_negatives + K.epsilon())


def total_loss(y_true, y_pred, num_classes):
    """
    CCE or BCE + Dice Loss + IOU Loss
    :param y_true:
    :param y_pred:
    :param num_classes:
    :return:
    """
    if num_classes == 1:
        ce_loss = categorical_crossentropy(y_true, y_pred)
    else:
        ce_loss = binary_crossentropy(y_true, y_pred)

    dice_loss = 1.0 - dice_coef(y_true, y_pred).numpy()
    iou_loss = 1.0 - iou_metric(y_true, y_pred).numpy / ()

    TOTAL_LOSS = ce_loss + dice_loss + iou_loss

    return TOTAL_LOSS


def loss_fn(y_true, y_pred, num_classes):
    return total_loss(y_true, y_pred, num_classes)


path = 'BestModels'
latestModels = sorted(os.listdir(path), reverse=True)
latestModelsList = [i for i in os.listdir(os.path.join(path, latestModels[0])) if i.endswith('.keras')]
modelDict = {i.split('__')[0]: i for i in latestModelsList}
modelKey = st.selectbox('Choose the Trained Model: ', modelDict.keys())
selectedModel = modelDict[modelKey]

model = tf.keras.models.load_model(filepath=os.path.join(path, latestModels[0], selectedModel),
                                   custom_objects={
                                       'loss_fn': loss_fn,
                                       'iou_metric': iou_metric,
                                       'dice_coef': dice_coef,
                                       'dice_coef_necrotic': dice_coef_necrotic,
                                       'dice_coef_edema': dice_coef_necrotic,
                                       'dice_coef_enhancing': dice_coef_necrotic,
                                       'dice_coef_no_tumour': dice_coef_no_tumour,
                                       'precision': precision,
                                       'sensitivity': sensitivity,
                                       'specificity': specificity
                                   })

dataType = st.radio('Choose dataset type: ', ['Test', 'Val', 'Train'], horizontal=True, index=0)
dataPath = 'BrainTS/training/test/images'.replace("test/", dataType.lower() + '/')


def evaluation_metrics(path, type):
    # List all image files in the directory
    imgPaths = [os.path.join(path, i) for i in os.listdir(path)]

    # Initialize a progress bar
    progress_bar = st.progress(0)

    # Calculate the total number of images to be processed
    total_images = len(imgPaths)
    #st.write(total_images)

    # Initialize a list to store the metrics
    METRICS = []

    for idx, img_path in enumerate(imgPaths):
        img = np.load(img_path)
        msk = np.load(img_path.replace('image', 'mask'))
        img_exp = np.expand_dims(img, axis=0)
        pred_msk = model.predict(img_exp, verbose=0)[0]

        pred_msk[pred_msk >= thresh] = 1
        pred_msk[pred_msk < thresh] = 0

        METRICS.append(
            (iou_metric(msk, pred_msk).numpy(),
             precision(msk, pred_msk).numpy(),
             sensitivity(msk, pred_msk).numpy(),
             specificity(msk, pred_msk).numpy(),
             dice_coef(msk, pred_msk).numpy(),
             dice_coef_necrotic(msk, pred_msk).numpy(),
             dice_coef_edema(msk, pred_msk).numpy(),
             dice_coef_enhancing(msk, pred_msk).numpy(),
             dice_coef_enhancing(msk, pred_msk).numpy(),
             dice_coef_no_tumour(msk, pred_msk).numpy())
        )

        # Update the progress bar (calculate progress as a percentage)
        progress = (idx + 1) / total_images  # Current progress
        progress_bar.progress(progress)  # Update the progress bar

    return METRICS


metrics = evaluation_metrics(dataPath, dataType.upper())

mean_metrics = np.mean(metrics, axis=0)

metrics_df = pd.DataFrame([
    ["IOU_METRIC", mean_metrics[0]],
    ["PRECISION", mean_metrics[1]],
    ["SENSITIVITY", mean_metrics[2]],
    ["SPECIFICITY", mean_metrics[3]],
    ["DICE_COEF", mean_metrics[4]],
    ["DICE_COEF_NECROTIC", mean_metrics[5]],
    ["DICE_COEF_EDEMA", mean_metrics[6]],
    ["DICE_COEF_ENHANCING", mean_metrics[7]],
    ["DICE_COEF_NO_TUMOUR", mean_metrics[8]]

], columns=['METRIC', 'VALUE'])

st.table(metrics_df)
