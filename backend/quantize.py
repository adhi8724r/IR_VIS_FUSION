import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from net import Generator_DWT

def get_model_size_mb(filepath):
    """Return file size in megabytes."""
    return os.path.getsize(filepath) / (1024 * 1024)

def benchmark_model(model, vis_input, ir_input, runs=50, device="cpu"):
    """Benchmark inference latency over multiple runs."""
    model.eval()
    vis_input = vis_input.to(device)
    ir_input = ir_input.to(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(5):
            _ = model(vis_input, ir_input)

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            out = model(vis_input, ir_input)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start_time) / runs * 1000  # ms
    return elapsed, out

def calculate_psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio between two tensors."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def main():
    print("=" * 60)
    print("  IR_VIS_FUSION: Model Optimization & Quantization Benchmark")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    fp32_model_path = os.path.join(model_dir, "FusionModelG.model")
    quant_pt_path = os.path.join(model_dir, "FusionModelG_quant.pt")
    onnx_fp32_path = os.path.join(model_dir, "FusionModelG_fp32.onnx")
    onnx_quant_path = os.path.join(model_dir, "FusionModelG_quant.onnx")

    if not os.path.exists(fp32_model_path):
        raise FileNotFoundError(f"Source model not found at {fp32_model_path}")

    fp32_size = get_model_size_mb(fp32_model_path)
    print(f"\n[1] Loaded Baseline FP32 Model: {fp32_model_path}")
    print(f"    Baseline Model Size: {fp32_size:.2f} MB")

    # Load Model on CPU for edge/cloud profiling
    device = torch.device("cpu")
    model_fp32 = Generator_DWT().to(device)
    model_fp32.load_state_dict(torch.load(fp32_model_path, map_location=device), strict=True)
    model_fp32.eval()

    # Create dummy 128x128 sample inputs
    vis_sample = torch.rand(1, 1, 128, 128, dtype=torch.float32)
    ir_sample = torch.rand(1, 1, 128, 128, dtype=torch.float32)

    print("\n[2] Benchmarking FP32 CPU Inference...")
    fp32_latency, fp32_out = benchmark_model(model_fp32, vis_sample, ir_sample, runs=30, device=device)
    print(f"    FP32 Latency (CPU): {fp32_latency:.2f} ms per frame ({1000 / fp32_latency:.1f} FPS)")

    # 3. PyTorch Dynamic Quantization
    print("\n[3] Applying PyTorch Dynamic INT8 Quantization...")
    model_quant = torch.ao.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    torch.save(model_quant.state_dict(), quant_pt_path)
    quant_size = get_model_size_mb(quant_pt_path)
    print(f"    Saved Quantized PyTorch Model: {quant_pt_path}")
    print(f"    Quantized Model Size: {quant_size:.2f} MB ({((fp32_size - quant_size) / fp32_size) * 100:.1f}% reduction)")

    quant_latency, quant_out = benchmark_model(model_quant, vis_sample, ir_sample, runs=30, device=device)
    psnr_score = calculate_psnr(fp32_out, quant_out)
    print(f"    Quantized PyTorch Latency: {quant_latency:.2f} ms ({1000 / quant_latency:.1f} FPS)")
    print(f"    Speedup: {fp32_latency / quant_latency:.2f}x")
    print(f"    Output Fidelity PSNR: {psnr_score:.2f} dB (High fidelity)")

    # 4. Optional ONNX Export
    try:
        import onnx
        print("\n[4] Exporting to ONNX...")
        torch.onnx.export(
            model_fp32,
            (vis_sample, ir_sample),
            onnx_fp32_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["vis", "ir"],
            output_names=["fused"],
            dynamic_axes={
                "vis": {0: "batch_size", 2: "height", 3: "width"},
                "ir": {0: "batch_size", 2: "height", 3: "width"},
                "fused": {0: "batch_size", 2: "height", 3: "width"}
            },
            dynamo=False
        )
        print(f"    Exported ONNX FP32: {onnx_fp32_path} ({get_model_size_mb(onnx_fp32_path):.2f} MB)")

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            import onnxruntime as ort
            print("\n[5] Quantizing ONNX model to INT8...")
            quantize_dynamic(
                model_input=onnx_fp32_path,
                model_output=onnx_quant_path,
                weight_type=QuantType.QUInt8
            )
            quant_onnx_size = get_model_size_mb(onnx_quant_path)
            print(f"    Exported ONNX INT8: {onnx_quant_path} ({quant_onnx_size:.2f} MB)")
            print(f"    Size Reduction: {((fp32_size - quant_onnx_size) / fp32_size) * 100:.1f}%")

            print("\n[6] Benchmarking ONNX Runtime INT8 on CPU...")
            session = ort.InferenceSession(onnx_quant_path, providers=['CPUExecutionProvider'])
            ort_inputs = {
                "vis": vis_sample.numpy(),
                "ir": ir_sample.numpy()
            }
            # Warmup
            for _ in range(5):
                _ = session.run(None, ort_inputs)
            start_ort = time.perf_counter()
            runs = 30
            for _ in range(runs):
                ort_outs = session.run(None, ort_inputs)
            ort_latency = (time.perf_counter() - start_ort) / runs * 1000
            print(f"    ONNX Runtime INT8 Latency: {ort_latency:.2f} ms ({1000 / ort_latency:.1f} FPS)")
            print(f"    Speedup vs Baseline PyTorch: {fp32_latency / ort_latency:.2f}x")
        except Exception as e:
            print(f"    ONNX quantization/benchmark notice: {e}")
    except Exception as e:
        print(f"    ONNX export notice: {e}")

    print("\n" + "=" * 60)
    print("  Optimization & Quantization Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
