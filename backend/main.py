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

import onnxruntime as ort
import psutil
import gc

from contextlib import asynccontextmanager

process = psutil.Process(os.getpid())

def memory_usage(stage: str):
    ram = process.memory_info().rss / (1024 ** 2)
    msg = f"[{stage}] RAM: {ram:.2f} MB"
    if torch.cuda.is_available():
        gpu = torch.cuda.memory_allocated() / (1024 ** 2)
        msg += f" | GPU: {gpu:.2f} MB"
    print(msg)

fusion_engine = None
use_onnx = False
yolo_model = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
onnx_model_path = os.path.join(BASE_DIR, "models", "FusionModelG_quant.onnx")
pytorch_model_path = os.path.join(BASE_DIR, "models", "FusionModelG.model")

def load_pytorch_model(model_path):
    model = Generator_DWT().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    return model

def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        print("Lazy-loading YOLOv11 model on demand...")
        memory_usage("Before loading YOLO")
        yolo_model_path = os.path.join(BASE_DIR, "yolo11n.pt")
        yolo_model = ultralytics.YOLO(yolo_model_path)
        memory_usage("After loading YOLO")
    return yolo_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    global fusion_engine, use_onnx

    memory_usage("Startup - Init")

    if os.path.exists(onnx_model_path):
        print(f"Loading Quantized ONNX Fusion model from: {onnx_model_path}")
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 1
        
        # Explicit CPUExecutionProvider avoids DRM GPU probe errors on cloud containers
        providers = ['CPUExecutionProvider']
        fusion_engine = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=providers)
        use_onnx = True
        print(f"ONNX Fusion model successfully loaded with providers: {fusion_engine.get_providers()}")
    else:
        print(f"Loading PyTorch Fusion model from: {pytorch_model_path}")
        fusion_engine = load_pytorch_model(pytorch_model_path)
        use_onnx = False

    memory_usage("Startup - After loading fusion model")
    print("Backend started successfully! (YOLO will lazy-load on /detect)")

    yield
    print("shutting down")

app = FastAPI(title="Image Fusion and object detection", lifespan=lifespan)

# Configure CORS for local development and Vercel production
cors_origins_str = os.environ.get("CORS_ORIGINS", "")
if cors_origins_str:
    origins = [orig.strip() for orig in cors_origins_str.split(",") if orig.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
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
    return {
        'message': 'Image Fusion App',
        'engine': 'ONNX Runtime INT8 (Quantized)' if use_onnx else 'PyTorch FP32',
        'model_file': 'FusionModelG_quant.onnx' if use_onnx else 'FusionModelG.model'
    }

@app.get('/memory')
def get_memory():
    ram = process.memory_info().rss / (1024 ** 2)
    data = {"ram_mb": round(ram, 2)}
    if torch.cuda.is_available():
        data["gpu_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
    return data

# Fusion endpoint
@app.post('/fuse')
async def fuse_images(
        vis : UploadFile = File(...),
        ir : UploadFile = File(...)
):
    memory_usage("Before preprocessing (fuse)")
    
    # 1. SETUP SAVE LOCATION
    save_dir = os.path.join(BASE_DIR, "fused_results")
    os.makedirs(save_dir, exist_ok=True)

    vis_img = Image.open(io.BytesIO(await vis.read()))
    ir_img = Image.open(io.BytesIO(await ir.read())).convert('L')

    # Cap max dimension to 640px to prevent Out-Of-Memory (OOM 502) on free cloud containers (512MB RAM limit)
    MAX_DIM = 640
    w, h = vis_img.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        vis_img = vis_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

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

    memory_usage("After preprocessing (fuse)")

    # Run inference using ONNX Runtime or PyTorch
    if use_onnx:
        vis_in = (Y_pad.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
        ir_in = (ir_pad.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
        fused_raw = fusion_engine.run(None, {"vis": vis_in, "ir": ir_in})[0]
        Y_fused_pad = (np.clip(fused_raw, 0.0, 1.0) * 255.0).squeeze().astype(np.uint8)
    else:
        vis_t = torch.from_numpy(Y_pad).unsqueeze(0).unsqueeze(0).float() / 255.0
        ir_t  = torch.from_numpy(ir_pad).unsqueeze(0).unsqueeze(0).float() / 255.0
        vis_t = vis_t.to(device)
        ir_t  = ir_t.to(device)

        with torch.inference_mode():
            fused_t = fusion_engine(vis_t, ir_t)

        Y_fused_pad = tensor_to_uint8(fused_t)

    memory_usage("After fusion")

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
    
    out_pil = Image.fromarray(fused_out)
    out_pil.save(save_path)
    print(f"Saved fused image to: {save_path}")

    # Prepare response image buffer
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    buf.seek(0)
    image_bytes = buf.read()

    memory_usage("After saving (fuse)")

    # Clean temporary buffers
    del vis_np, ir_np, Y_pad, ir_pad, fused_out, out_pil, buf
    gc.collect()
    memory_usage("After GC cleanup (fuse)")

    return Response(
        content=image_bytes,
        media_type="image/png",
        status_code=200
    )

# Object detection endpoint
@app.post('/detect', status_code=200)
async def yolo_detect(
    vis: UploadFile = File(...),
    ir: UploadFile = File(...),
    fus: UploadFile = File(...)
):
    memory_usage("Before preprocessing (detect)")

    raw_inputs = [
        ('vis', await vis.read()),
        ('ir', await ir.read()),
        ('fus', await fus.read())
    ]

    keys = []
    pil_imgs = []
    MAX_DIM = 640

    for key, data in raw_inputs:
        img = Image.open(io.BytesIO(data)).convert('RGB')
        w, h = img.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        keys.append(key)
        pil_imgs.append(img)

    memory_usage("After preprocessing (detect)")

    model = get_yolo_model()

    # Batch prediction with imgsz=320 to guarantee fast execution within Cloudflare 50s timeout and low memory
    with torch.inference_mode():
        predictions = model.predict(pil_imgs, imgsz=320, device='cpu', verbose=False)

    memory_usage("After YOLO")

    results_b64 = {}
    for key, res in zip(keys, predictions):
        plotted = res.plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        results_b64[key] = img_2_b64(plotted_rgb)

    memory_usage("After saving/encoding (detect)")

    # Clean temporary prediction & image buffers
    del raw_inputs, pil_imgs, predictions
    gc.collect()
    memory_usage("After GC cleanup (detect)")

    return results_b64

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
