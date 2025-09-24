import os
import sys
import argparse
import yaml
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from ultralytics import YOLO
from pytorch_nndct.apis import torch_quantizer

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# ---------------------------------------
# Arguments
# ---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_yaml', type=str, required=True, help='Path to data.yaml (YOLO dataset)')
parser.add_argument('--weights', type=str, required=True, help='YOLO model weights .pt file (e.g., yolo11n.pt)')
parser.add_argument('--model_out_dir', type=str, default='./quant_results', help='Output folder for results')
parser.add_argument('--config_file', default=None, help='Quantization config file (optional)')
parser.add_argument('--quant_mode', default='calib', choices=['float', 'calib', 'test'], help='float/calib/test')
parser.add_argument('--fast_finetune', action='store_true', help='Run fast finetune before calibration')
parser.add_argument('--deploy', action='store_true', help='Export xmodel (only in test mode)')
parser.add_argument('--inspect', action='store_true', help='Run inspector mode (requires target)')
parser.add_argument('--target', default=None, help='Target device (e.g., DPUCZDX8G_ISA)')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--subset_len', type=int, default=None, help='Subset length for calibration/evaluation')
parser.add_argument('--img_size', type=int, default=640, help='Image size (square) used by YOLO')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.model_out_dir, exist_ok=True)

# ---------------------------------------
# Dataset
# ---------------------------------------
class Yolov5LikeDataset(Dataset):
    def __init__(self, imgs_list, img_size=640, transform=None, max_samples=None):
        self.imgs = imgs_list
        if max_samples:
            self.imgs = self.imgs[:max_samples]
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        p = self.imgs[idx]
        img = Image.open(p).convert('RGB')
        img = self.transform(img)
        return img, p

# ---------------------------------------
# Parse data.yaml
# ---------------------------------------
def parse_data_yaml(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    def expand_paths(entry):
        if isinstance(entry, list):
            return entry
        if isinstance(entry, str):
            p = Path(entry)
            if p.is_dir():
                exts = {'.jpg', '.jpeg', '.png', '.bmp'}
                imgs = [str(x) for x in p.rglob('*') if x.suffix.lower() in exts]
                imgs.sort()
                return imgs
            elif p.is_file():
                if p.suffix.lower() == '.txt':
                    lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
                    return lines
                elif p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    return [str(p)]
        return []

    train_list = expand_paths(data.get('train', []))
    val_list   = expand_paths(data.get('val', [])) or train_list
    names      = data.get('names', None)
    return train_list, val_list, names

# ---------------------------------------
# Simple inference function
# ---------------------------------------
def run_inference_and_count(model_torch, data_loader, device):
    model_torch.eval()
    model_torch.to(device)
    total_images, total_detections = 0, 0

    with torch.no_grad():
        for imgs, _ in tqdm(data_loader, total=len(data_loader), desc='Infer'):
            imgs = imgs.to(device)
            out = model_torch(imgs)

            if torch.is_tensor(out):
                flat = out.view(out.size(0), -1)
                act = (flat.abs().sum(dim=1) > 0).cpu().int()
                total_detections += int(act.sum().item())
            elif isinstance(out, (list, tuple)):
                candidate = out[0]
                if torch.is_tensor(candidate):
                    det = (candidate.abs().view(candidate.size(0), -1).sum(dim=1) > 0).cpu().int()
                    total_detections += int(det.sum().item())
                else:
                    total_detections += len(out)

            total_images += imgs.size(0)

    avg_det_per_image = total_detections / max(1, total_images)
    return total_images, total_detections, avg_det_per_image

# ---------------------------------------
# Main function
# ---------------------------------------
def quantize_yolo():
    train_list, val_list, _ = parse_data_yaml(args.data_yaml)
    if len(val_list) == 0:
        raise RuntimeError("Validation set (val) not found in data.yaml")

    if args.subset_len:
        random.shuffle(val_list)
        val_list = val_list[:args.subset_len]

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    val_dataset = Yolov5LikeDataset(val_list, img_size=args.img_size, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Load YOLO model
    print("[INFO] Loading YOLO:", args.weights)
    yolo = YOLO(args.weights)
    base_model = yolo.model
    base_model.cpu()
    base_model.eval()

    sample_input = torch.randn([args.batch_size, 3, args.img_size, args.img_size])

    if args.quant_mode == 'float':
        quant_model = base_model
    else:
        print(f"[INFO] Creating quantizer: {args.quant_mode}")
        quantizer = torch_quantizer(
            args.quant_mode,
            base_model,
            (sample_input,),
            device=device,
            quant_config_file=args.config_file,
            target=args.target
        )
        quant_model = quantizer.quant_model

    print("[INFO] Running quantization test...")
    total_imgs, total_dets, avg = run_inference_and_count(quant_model, val_loader, device)
    print(f"[RESULT] Images={total_imgs}, Detections={total_dets}, Avg per image={avg:.3f}")

    if args.quant_mode == 'calib':
        print("[INFO] Exporting quantization configuration...")
        quantizer.export_quant_config()

    if args.deploy and args.quant_mode == 'test':
        print("[INFO] Exporting models (torchscript/onnx/xmodel)...")
        quantizer.export_torch_script(os.path.join(args.model_out_dir, "quant_model_ts.pt"))
        quantizer.export_onnx_model(os.path.join(args.model_out_dir, "quant_model.onnx"))
        quantizer.export_xmodel(deploy_check=False)

    try:
        out_pt = os.path.join(args.model_out_dir, "quant_model_final.pt")
        torch.save(quant_model.state_dict(), out_pt)
        print(f"[INFO] Saved quant_model: {out_pt}")
    except Exception as e:
        print("[WARN] Failed to save quant_model:", e)

    print("[INFO] Quantization finished.")

if __name__ == '__main__':
    quantize_yolo()
