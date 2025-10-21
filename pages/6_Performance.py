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

st.set_page_config(layout="wide")

SMOOTH = 1e-6



path = 'BestModels'
latestModels = sorted(os.listdir(path), reverse=True)
latestModelsListHistory = [i for i in os.listdir(os.path.join(path, latestModels[0])) if i.endswith('.txt')]
modelHistDict = {i.split('_')[0]: i for i in latestModelsListHistory}
modelHistKey = st.selectbox('Choose the Trained Model: ', modelHistDict.keys())
selectedModelHistory = modelHistDict[modelHistKey]


# st.write(selectedModel)
# st.write(os.path.join(os.path.join(path, latestModels[0]), selectedModel))
history = pd.DataFrame(json.loads(open(os.path.join(os.path.join(path, latestModels[0]), selectedModelHistory)).read()))
# st.write(history)


fig, axes = plt.subplots(5, 2, figsize=(25, 40))

axes[0][0].plot(history['accuracy'], label='Accuracy')
axes[0][0].plot(history['val_accuracy'], label='Val_Accuracy')
axes[0][0].set_title('ACCURACY')
axes[0][0].legend()

axes[0][1].plot(history['loss'], label='Loss')
axes[0][1].plot(history['val_loss'], label='Val_Loss')
axes[0][1].set_title('LOSS')
axes[0][1].legend()

axes[1][0].plot(history['dice_coef'], label='Dice_Coeff')
axes[1][0].plot(history['val_dice_coef'], label='Val_Dice_Coeff')
axes[1][0].set_title('DICE_COEF')
axes[1][0].legend()

axes[1][1].plot(history['precision'], label='Precision')
axes[1][1].plot(history['val_precision'], label='Val_Precision')
axes[1][1].set_title('Precision')
axes[1][1].legend()

axes[2][0].plot(history['dice_coef_necrotic'], label='Dice_Coeff_Necrotic')
axes[2][0].plot(history['dice_coef_necrotic'], label='Val_Dice_Coef_Necrotic')
axes[2][0].set_title('DICE_COEF_NECROTIC')
axes[2][0].legend()

axes[2][1].plot(history['dice_coef_edema'], label='Dice_Coef_Edema')
axes[2][1].plot(history['val_dice_coef_edema'], label='Val_Dice_Coef_Edema')
axes[2][1].set_title('DICE_COEF_EDEMA')
axes[2][1].legend()

axes[3][0].plot(history['dice_coef_enhancing'], label='Dice_Coef_Enhancing')
axes[3][0].plot(history['val_dice_coef_enhancing'], label='Val_Dice_Coef_Enhancing')
axes[3][0].set_title('DICE_COEF_ENHANCING')
axes[3][0].legend()

axes[3][1].plot(history['iou_metric'], label='IOU_Metric')
axes[3][1].plot(history['val_iou_metric'], label='Val_IOU_Metric')
axes[3][1].set_title('IOU')
axes[3][1].legend()

# axes[3][1].plot(history['precision'], label='Precision')
# axes[3][1].plot(history['val_precision'], label='Val_Precision')
# axes[3][1].set_title('Precision')
# axes[3][1].legend()

axes[4][0].plot(history['sensitivity'], label='Sensitivity')
axes[4][0].plot(history['val_sensitivity'], label='Val_Sensitivity')
axes[4][0].set_title('Sensitivity')
axes[4][0].legend()

axes[4][1].plot(history['specificity'], label='Specificity')
axes[4][1].plot(history['val_specificity'], label='Val_Specificity')
axes[4][1].set_title('Specificity')
axes[4][1].legend()


plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.35)

st.pyplot(fig)
