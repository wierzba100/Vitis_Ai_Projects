import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO
from pytorch_nndct.apis import torch_quantizer, Inspector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# Arguments
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="Models/fine_tuned_yolov8s.pt", help="Path to YOLOv8 model")
parser.add_argument("--data_dir", default="./calib_data", help="Folder with calibration images")
parser.add_argument("--quant_mode", default="calib", choices=["float", "calib", "test"], help="Quantization mode")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--subset_len", default=100, type=int, help="How many images to use for calibration/test")
parser.add_argument("--deploy", action="store_true", help="Export .xmodel")
parser.add_argument("--inspect", action="store_true", help="Run Inspector for target check")
parser.add_argument("--target", default="DPUCZDX8G_ISA1_B4096", help="Target device")
args, _ = parser.parse_known_args()

# ======================================================
# Calibration Dataset
# ======================================================
class ImageFolderDataset(Dataset):
    def __init__(self, folder, size=640, max_len=100):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
        self.size = size
        self.max_len = max_len

    def __len__(self):
        return min(len(self.files), self.max_len)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = img.resize((self.size, self.size))
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img  # [3,H,W], DataLoader will add batch

# ======================================================
# Replace SiLU → SiLU_custom
# ======================================================
class SiLU_custom(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def replace_silu(model):
    for name, module in model.named_children():
        if isinstance(module, nn.SiLU):
            setattr(model, name, SiLU_custom())
        else:
            replace_silu(module)
    return model

# ======================================================
# Fix Detect (remove warnings "is not tensor")
# ======================================================
def freeze_detect(model):
    for m in model.modules():
        if m.__class__.__name__ == "Detect":
            if isinstance(m.stride, (list, tuple)):
                m.stride = torch.tensor(m.stride, dtype=torch.float32)
            if isinstance(m.anchors, (list, tuple)):
                m.anchors = torch.tensor(m.anchors, dtype=torch.float32)
            if hasattr(m, "names"):
                delattr(m, "names")
    return model

# ======================================================
# Quantization
# ======================================================
def quantization():
    # 1. Load YOLOv8
    yolomodel = YOLO(args.model_path).model
    yolomodel = replace_silu(yolomodel)
    yolomodel = freeze_detect(yolomodel)
    yolomodel.eval().cpu()

    dummy_input = torch.randn([1, 3, 640, 640])

    # 2. Inspector
    if args.quant_mode == "float":
        if args.inspect:
            inspector = Inspector(args.target)
            inspector.inspect(yolomodel, (dummy_input,), device=device)
            sys.exit()
        return

    # 3. Quantization
    quantizer = torch_quantizer(
        args.quant_mode,
        yolomodel,
        (dummy_input,),
        device=device,
        target=args.target
    )
    quant_model = quantizer.quant_model

    # 4. Calibration/Test
    dataset = ImageFolderDataset(args.data_dir, max_len=args.subset_len)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Calibration")):
            batch = batch.to(device)             # [B,3,H,W]
            _ = quant_model(batch)               # forward
            if i >= args.subset_len:
                break

    # 5. Export
    if args.quant_mode == "calib":
        quantizer.export_quant_config()
    if args.deploy and args.quant_mode == "test":
        quantizer.export_xmodel()

    print("Quantization finished. Output files are in ./quantize_result/")

# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    quantization()
