from fastapi import FastAPI,UploadFile,File,Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import uvicorn
import base64
import ultralytics
import cv2
import torch
from net import Generator_DWT
import os
import time
from utils import tensor_to_uint8,pad_to_128,unpad

from contextlib import asynccontextmanager

fusion_model = None
yolo_model = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fusion_model_path = "models/FusionModelG.model"

def load_model(model_path):
    model = Generator_DWT().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    global fusion_model, yolo_model
    print("Fusion model is loading")
    fusion_model = load_model(fusion_model_path)

    print("yolo model is loading")
    yolo_model = ultralytics.YOLO("yolo11n.pt")
    
    yield
    print("shutting down")

app = FastAPI(title="Image Fusion and object detection",lifespan=lifespan)

# Configure CORS
origins = [
    "http://localhost:5173",  # React-Vite development server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def img_2_b64(img_np):
    img_pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.get('/')
def main():
    return {'message': 'Image Fusion App'}

# Fusion endpoint
@app.post('/fuse')
async def fuse_images(
        vis : UploadFile = File(...),
        ir : UploadFile = File(...)
):
    
    # 1. SETUP SAVE LOCATION
    save_dir = "fused_results"
    os.makedirs(save_dir, exist_ok=True)

    vis_img = Image.open(io.BytesIO(await vis.read()))
    ir_img = Image.open(io.BytesIO(await ir.read())).convert('L')

    vis_np = np.array(vis_img)
    ir_np = np.array(ir_img)

    ir_np = cv2.resize(
        ir_np,
        (vis_np.shape[1], vis_np.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    
    orig_h, orig_w = vis_np.shape[:2]    

    # If VIS is RGB → convert to YCrCb and extract Y
    if vis_np.ndim == 3:
        vis_ycrcb = cv2.cvtColor(vis_np, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(vis_ycrcb)
    else:
        Y = vis_np
        Cr, Cb = None, None

    Y_pad, h0, w0 = pad_to_128(Y)
    ir_pad, _, _  = pad_to_128(ir_np)

    vis_t = torch.from_numpy(Y_pad).unsqueeze(0).unsqueeze(0).float() / 255.0
    ir_t  = torch.from_numpy(ir_pad).unsqueeze(0).unsqueeze(0).float() / 255.0

    vis_t = vis_t.to(device)
    ir_t  = ir_t.to(device)

    # Inference
    with torch.no_grad():
        fused_t = fusion_model(vis_t, ir_t)

    # Postprocess
    Y_fused_pad = tensor_to_uint8(fused_t)

    # Remove padding
    Y_fused = unpad(Y_fused_pad,h0,w0) 

    # If original VIS was RGB → reconstruct RGB
    if Cr is not None:
        fused_ycrcb = cv2.merge([Y_fused, Cr, Cb])
        fused_rgb   = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2RGB)
        
        fused_out = cv2.resize(
            fused_rgb,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        fused_out = cv2.resize(
            Y_fused,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR
        )

    timestamp = int(time.time())
    filename = f"fused_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    
    Image.fromarray(fused_out).save(save_path)
    print(f"Saved fused image to: {save_path}")

    # Return image
    buf = io.BytesIO()
    Image.fromarray(fused_out).save(buf, format="PNG")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="image/png",
        status_code=200
    )

#Object detection endpoint
@app.post('/detect',status_code=200)
async def yolo_detect(
    vis : UploadFile = File(...),
    ir : UploadFile = File(...),
    fus: UploadFile = File(...)
):
    inputs = {'vis': vis, 'ir': ir, 'fus': fus}

    results_b64 = {}

    for key,img_bytes in inputs.items():
        img = Image.open(io.BytesIO(await img_bytes.read())).convert('RGB')

        res = yolo_model.predict(img)
        plotted = res[0].plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        results_b64[key] = img_2_b64(plotted_rgb)

    return results_b64

if __name__=="__main__":
    uvicorn.run('main:app',host='localhost',port=8000,reload=True)
