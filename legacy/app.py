import base64

import streamlit as st
import requests
from PIL import Image
import io
import base64

st.title("Image Fusion")

FUSE_URL = "http://localhost:8000/fuse"
DETECT_URL = "http://localhost:8000/detect"

if 'fused_bytes' not in st.session_state:
    st.session_state.fused_bytes = None

vis_file = st.file_uploader(
    "upload an visible image",
    type=['png','jpg','jpeg','bmp']
)

ir_file = st.file_uploader(
    "upload an IR image",
    type=['png','jpg','jpeg','bmp']
)

def base64_2_image(string):
    img_data = base64.b64decode(string)
    image = Image.open(io.BytesIO(img_data))
    return image

if vis_file and ir_file:
    vis_img = Image.open(vis_file)
    ir_img = Image.open(ir_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(vis_img,caption=f"Visible Image{vis_img.size}")
    
    with col2:
        st.image(ir_img,caption=f"Infrared Image{ir_img.size}")

    if st.button("Generate fused image"):
        if st.spinner("Fusing images"):
            vis_file.seek(0)
            ir_file.seek(0)

            files = {
                'vis' : vis_file.getvalue(),
                'ir' : ir_file.getvalue()
            }

            response = requests.post(FUSE_URL,files=files)

            if response.status_code==200:
                st.session_state.fused_bytes = response.content
            else:
                st.error('fusion failed')

if st.session_state.fused_bytes is not None:
    fus_img = Image.open(io.BytesIO(st.session_state.fused_bytes))
    left,center,right=st.columns([1,2,1])
    with center:
        st.image(fus_img, caption=f"Fused Output{fus_img.size}", width="content")

    if st.button("detect objects"):
        if st.spinner("object detection using yolo"):
            vis_file.seek(0)
            ir_file.seek(0)

            files = {
                'vis' : vis_file.getvalue(),
                'ir' : ir_file.getvalue(),
                'fus' : st.session_state.fused_bytes
            }

            response2 = requests.post(DETECT_URL, files=files)

            if response2.status_code==200:
                data = response2.json()

                res_vis = base64_2_image(data['vis'])
                res_ir = base64_2_image((data['ir']))
                res_fus = base64_2_image((data['fus']))

                d_col1, d_col2 = st.columns(2)

                with d_col1:
                    st.image(res_vis, caption="Object detection on Visible Image", width="content")

                with d_col2:
                    st.image(res_ir, caption="Object detection on Infrared Image", width="content")

                d_left, d_center, d_right = st.columns([1, 2, 1])
                with d_center:
                    st.image(res_fus, caption="Object detection on Fused Output", width="content")
            else:
                st.error('object detection failed')