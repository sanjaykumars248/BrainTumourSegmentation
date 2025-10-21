import os
import streamlit as st
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(layout="wide")


def load_data(pth, mask=False):
    sc = MinMaxScaler()
    img = nib.load(pth).get_fdata().T
    if mask:
        img[img == 4] = 3
    img = sc.fit_transform(img.reshape(-1, 1)).reshape(img.shape)
    img = img[13:141, 30:222, 24:216]
    return img


def plot_image_2d(path, train_type='Training'):
    imgs = [os.path.join(path, i) for i in os.listdir(path)]

    imgDict = {
        i.split('_')[-1].removesuffix('.nii').capitalize(): load_data(i) for i in imgs if 'seg' not in i
    }

    if train_type == 'Training':
        imgDict['Mask'] = load_data(str([i for i in imgs if 'seg' in i][0]), mask=True)
        fig = make_subplots(rows=1, cols=5, subplot_titles=("Flair", "T1", "T1ce", "T2", "Mask"))
    else:
        fig = make_subplots(rows=1, cols=4, subplot_titles=("Flair", "T1", "T1ce", "T2"))

    subplot_keys = list(imgDict.keys())

    # Get the number of slices in the 3D volumes
    num_slices = imgDict[subplot_keys[0]].shape[0]

    # Add initial frame (slice 64 or the middle slice)
    initial_slice = num_slices // 2

    posList = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]

    for idx, (row, col) in enumerate(posList[:len(subplot_keys)]):
        key = subplot_keys[idx]
        px_fig = px.imshow(imgDict[key][initial_slice, :, :], color_continuous_scale="Gray")  # Generate px.imshow figure
        trace = px_fig.data[0]  # Extract the heatmap trace
        fig.add_trace(trace, row=row, col=col)

    # Create frames for animation
    frames = []
    for slice_idx in range(0, num_slices, 4):
        frame_data = []
        for idx, (row, col) in enumerate(posList[:len(subplot_keys)]):
            key = subplot_keys[idx]
            slice_data = imgDict[key][slice_idx, :, :]
            frame_data.append(
                go.Heatmap(z=slice_data, colorscale="Gray", showscale=True)
            )
            frames.append(go.Frame(data=frame_data, name=str(slice_idx)))
    fig.frames = frames

    # Add slider for manual slice selection
    sliders = [
        {
            "steps": [
                {
                    "args": [[str(slice_idx)], {"frame": {"duration": 0}, "mode": "immediate"}],
                    "label": str(slice_idx),
                    "method": "animate",
                }
                for slice_idx in range(num_slices)
            ]
        }
    ]

    # Update layout
    fig.update_layout(
        title_text="MRI 2D Images by Modalities",
        coloraxis=dict(colorscale='viridis', showscale=True),
        sliders=sliders,

    )
    return fig


st.header('Brain MRI 2D')

train = st.radio('Choose dataset type: ', ['Training', 'Testing'], horizontal=True, index=0)
path = 'BraTS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/' if train == 'Training' \
    else 'BraTS20/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/'

col1, col2 = st.columns(2)

mriPath = st.selectbox('Select the sample Brain MRI file: ', sorted([i for i in os.listdir(path) if ".csv" not in i]))
imgPath = os.path.join(path, mriPath)

st.plotly_chart(plot_image_2d(imgPath, train))
