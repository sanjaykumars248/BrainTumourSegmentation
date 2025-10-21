import os
import streamlit as st
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

st.set_page_config(layout="wide")


def plot_image_3d(path):
    sc = MinMaxScaler()
    image = nib.load(path).get_fdata().T
    image = sc.fit_transform(image.reshape(-1, 1)).reshape(image.shape)
    image = image[13:141, 30:222, 24:216]
    rows, cols = image[0].shape
    nb_frames = image.shape[0]
    nb_frms = (nb_frames - 1) / 10

    frames = [
        go.Frame(
            data=go.Surface(
                z=(nb_frms - k * 0.1) * np.ones((rows, cols)),
                surfacecolor=np.flipud(image[nb_frames - 1 - k]),
                cmin=0, cmax=1
            ),
            name=str(k)
        ) for k in range(0, nb_frames, 3)
    ]

    initialFrame = go.Surface(
        z=nb_frms * np.ones((rows, cols)),
        surfacecolor=np.flipud(image[64]),
        colorscale='viridis',
        cmin=0, cmax=1,
        colorbar=dict(thickness=20, ticklen=4)
    )

    fig = go.Figure(
        frames=frames,
        data=[initialFrame],
        layout=go.Layout(
            title='MRI Image',
            height=640, width=640,
            scene=dict(
                zaxis=dict(range=[-0.1, nb_frms + 0.1], autorange=False),
                aspectratio=dict(x=1, y=1, z=1)
            )
        )
    )

    animation_settings = {

        "frame": {"duration": 50},
        "mode": "immediate",
        "transition": {"duration": 50, "easing": "linear"}

    }

    fig.update_layout(

        sliders=[{
            "steps": [{
                "args": [[frame.name], animation_settings],
                "label": str(k),
                "method": "animate"
            } for k, frame in enumerate(fig.frames)]
        }]

    )

    return fig


def load_data(pth, mask=False):
    sc = MinMaxScaler()
    img = nib.load(pth).get_fdata().T
    if mask:
        img[img == 4] = 3
    img = sc.fit_transform(img.reshape(-1, 1)).reshape(img.shape)
    img = img[13:141, 30:222, 24:216]
    return img


st.header('Brain MRI 3D')

train = st.radio('Choose dataset type: ', ['Training', 'Testing'], horizontal=True, index=0)
path = 'BraTS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/' if train == 'Training' \
    else 'BraTS20/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/'

col1, col2 = st.columns(2)

with col1:
    mriPath = st.selectbox('Select the sample Brain MRI file: ', sorted(os.listdir(path)))
    imgPath = os.path.join(path, mriPath)

with col2:
    fileTypes = [i.split('_')[-1].removesuffix('.nii').capitalize() for i in os.listdir(imgPath)]
    file = st.radio('Choose file type: ', fileTypes, horizontal=True, index=0)
    imagePath = os.path.join(path, mriPath, mriPath + '_' + file.lower() + '.nii')

left, middle, right = st.columns((2, 5, 2))
with middle:
    st.plotly_chart(plot_image_3d(imagePath))
