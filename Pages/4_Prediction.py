import os

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
import tensorflow as tf
from tensorflow.keras import backend as K
from config import config

st.set_page_config(layout="wide")

SMOOTH = 1e-6
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
    iou = (intersection + smooth) / (union + smooth)

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


# Dice Coefficient for No Tumour
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
    iou_loss = 1.0 - iou_metric(y_true, y_pred).numpy()

    TOTAL_LOSS = ce_loss + dice_loss + iou_loss

    return TOTAL_LOSS


def loss_fn(y_true, y_pred, num_classes):
    return total_loss(y_true, y_pred, num_classes)


def plot_results(img, msk, pred, train_type):

    # Display the shape of the input image
    # st.write("Image shape:", img.shape)

    # Create a figure and a 1x3 grid of subplots
    # if train_type == 'Train':
    fig, axes = plt.subplots(3, 5, figsize=(16, 12))
    # else:
    #     fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # Display the image on each subplot
    axes[0][0].imshow(img[:, :, :3])
    axes[0][0].set_title(f"Image - {train_type} - T2-Flair-T1CE")
    axes[0][0].axis('off')

    axes[0][1].imshow(img[:, :, 2])
    axes[0][1].set_title(f"Image - {train_type} - T2")
    axes[0][1].axis('off')

    axes[0][2].imshow(img[:, :, 0])
    axes[0][2].set_title(f"Image - {train_type} - Flair")
    axes[0][2].axis('off')

    axes[0][3].imshow(img[:, :, 1])
    axes[0][3].set_title(f"Image - {train_type} - T1CE")
    axes[0][3].axis('off')

    if img.shape[2] == 4:
        axes[0][4].imshow(img[:, :, 3])
        axes[0][4].set_title(f"Image - {train_type} - T1")
        axes[0][4].axis('off')
    else:
        fig.delaxes(axes[0][4])

    axes[1][0].imshow(msk[:, :, 1:])
    axes[1][0].set_title(f"Mask - {train_type} - RGB")
    axes[1][0].axis('off')

    axes[1][1].imshow(msk[:, :, 1], cmap='gray')
    axes[1][1].set_title(f"Mask - {train_type} - Tumour Core")  # necrotic and non-enhancing tumor core (NCR/NET — label 1)
    axes[1][1].axis('off')

    axes[1][2].imshow(msk[:, :, 2], cmap='gray')
    axes[1][2].set_title(f"Mask - {train_type} - Edema")  # Peritumoral edema (ED — label 2)
    axes[1][2].axis('off')

    axes[1][3].imshow(msk[:, :, 3], cmap='gray')
    axes[1][3].set_title(f"Mask - {train_type} - GDE Tumour")  # Gadolinium-Enhancing Tumor (ET — label 4)
    axes[1][3].axis('off')

    axes[1][4].imshow(msk[:, :, 0], cmap='gray_r')
    axes[1][4].set_title(f"Mask - {train_type} - No Tumour")
    axes[1][4].axis('off')



    axes[2][0].imshow(pred[:, :, 1:])
    axes[2][0].set_title(f"Pred_Mask - {train_type} - RGB")
    axes[2][0].axis('off')

    axes[2][1].imshow(pred[:, :, 1], cmap='gray')
    axes[2][1].set_title(f"Pred_Mask - {train_type} - Tumour Core")
    axes[2][1].axis('off')

    axes[2][2].imshow(pred[:, :, 2], cmap='gray')
    axes[2][2].set_title(f"Pred_Mask - {train_type} - Edema")
    axes[2][2].axis('off')

    axes[2][3].imshow(pred[:, :, 3], cmap='gray')
    axes[2][3].set_title(f"Pred_Mask - {train_type} - GDE Tumour")
    axes[2][3].axis('off')

    axes[2][4].imshow(pred[:, :, 0], cmap='gray_r')
    axes[2][4].set_title(f"Pred_Mask - {train_type} - No Tumour")
    axes[2][4].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # Display the figure using Streamlit
    # fig.delaxes(axes[0][0])
    st.pyplot(fig)


def model_prediction(model, path, file, train_type):
    basePath = path
    imagePath = basePath + '/images/'
    img = np.load(os.path.join(imagePath, file))
    msk = np.zeros_like(img)
    mskPath = str(os.path.join(imagePath, file)).replace('image', 'mask')
    msk = np.load(mskPath)
    img_ex = np.expand_dims(img, axis=0)
    y_pred = model.predict(img_ex)[0]

    # st.write(np.unique(y_pred[:,:,1]))

    # y_pred[y_pred >= thresh] = 1
    # y_pred[y_pred < thresh] = 0

    plot_results(img, msk, y_pred, train_type)

    df = pd.DataFrame(
        data=[['Whole Tumour', 'C-All', dice_coef(msk, y_pred), precision(msk, y_pred)],
              ['Tumour Core', 'C1', dice_coef(msk[:, :, 1], y_pred[:, :, 1]), precision(msk[:, :, 1], y_pred[:, :, 1])],
              ['Peritumoural Edeuma', 'C2', dice_coef(msk[:, :, 2], y_pred[:, :, 2]), precision(msk[:, :, 2], y_pred[:, :, 2])],
              ['GD Enhancing Tumour', 'C3', dice_coef(msk[:, :, 3], y_pred[:, :, 3]), precision(msk[:, :, 3], y_pred[:, :, 3])],
              ['No Tumour', 'C0', dice_coef(msk[:, :, 0], y_pred[:, :, 0]), precision(msk[:, :, 0], y_pred[:, :, 0])]
        ], columns=['Tumour Type', 'Class Label', 'DICE', 'PRECISION'])

    st.table(df)

    pass


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
# path = 'BrainTS/train/input3ch' if train == 'Train' else 'BrainTS/test/input3ch'
dataPath = 'BrainTS/training/test/'.replace("test/", dataType.lower()+'/')
file = st.selectbox('Select the sample Brain MRI file: ', sorted(os.listdir(dataPath + '/images')))

model_prediction(model, dataPath, file, dataType)
