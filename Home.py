import streamlit as st

st.set_page_config(
    page_title="Brain Tumour Segmentation",
)

st.markdown("<h1 style='text-align: center;'>RESNET ENCODER-BASED UNET FOR BRAIN TUMOUR SEGMENTATION</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZmY5aDZzYXY0NWkzdmgycTF3b20wOGY3dnRhdzlkOGR4eWM1c3l1OCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/encB7tn3Jt2qPG57xP/giphy.gif" 
             width="720" height="400">
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.success(
    'Select a page above'
)