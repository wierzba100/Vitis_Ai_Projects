import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from pytorch_nndct.apis import torch_quantizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================
# Argumenty
# ======================================
parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default="./data",
    help='Dataset path (MNIST).'
)
parser.add_argument(
    '--model_dir',
    default="./",
    help='Path to model .pth file.'
)
parser.add_argument(
    '--config_file',
    default=None,
    help='Quantization configuration file.'
)
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='Subset length for evaluation (default: full test set).'
)
parser.add_argument(
    '--batch_size',
    default=64,
    type=int,
    help='Batch size for evaluation.'
)
parser.add_argument('--quant_mode',
    default='calib',
    choices=['float', 'calib', 'test'],
    help='Quantization mode: float, calib, or test.'
)
parser.add_argument('--fast_finetune',
    dest='fast_finetune',
    action='store_true',
    help='Fast finetuning before calibration.'
)
parser.add_argument('--deploy',
    dest='deploy',
    action='store_true',
    help='Export xmodel for deployment.'
)
parser.add_argument('--inspect',
    dest='inspect',
    action='store_true',
    help='Run inspector mode.'
)
parser.add_argument('--target',
    dest='target',
    nargs="?",
    const="",
    help='Target device (e.g. DPU).'
)

args, _ = parser.parse_known_args()

# ======================================
# Definicja modelu CNN (MNIST)
# ======================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

# ======================================
# Ladowanie danych MNIST
# ======================================
def load_data(train=True, data_dir="./data", batch_size=64, subset_len=None, sample_method='random', **kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)

    if subset_len:
        assert subset_len <= len(dataset)
        if sample_method == 'random':
            dataset = Subset(dataset, random.sample(range(0, len(dataset)), subset_len))
        else:
            dataset = Subset(dataset, list(range(subset_len)))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, **kwargs)
    return loader

# ======================================
# Klasa do liczenia srednich wartosci
# ======================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# ======================================
# Funkcja do liczenia dokladnosci
# ======================================
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ======================================
# Ewaluacja modelu
# ======================================
def evaluate(model, val_loader, loss_fn):
    model.eval()
    model = model.to(device)
    top1 = AverageMeter('Acc@1', ':6.2f')
    total_loss = 0
    total_samples = 0
    for _, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        acc1, = accuracy(outputs, labels, topk=(1,))
        top1.update(acc1[0], images.size(0))
    return top1.avg, total_loss / total_samples

# ======================================
# Glowna funkcja kwantyzacji
# ======================================
def quantization(title='optimize', file_path=''):
    data_dir = args.data_dir
    quant_mode = args.quant_mode
    finetune = args.fast_finetune
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    inspect = args.inspect
    config_file = args.config_file
    target = args.target

    if quant_mode != 'test' and deploy:
        deploy = False
        print("Exporting xmodel works only in test mode disabled for this run.")

    model = Net().cpu()
    model.load_state_dict(torch.load(file_path, map_location="cpu"))

    input = torch.randn([batch_size, 1, 28, 28])

    if quant_mode == 'float':
        quant_model = model
        if inspect:
            if not target:
                raise RuntimeError("Target device must be specified for inspector.")
            from pytorch_nndct.apis import Inspector
            inspector = Inspector(target)
            inspector.inspect(quant_model, (input,), device=device)
            sys.exit()
    else:
        quantizer = torch_quantizer(
            quant_mode, model, (input,), device=device,
            quant_config_file=config_file, target=target
        )
        quant_model = quantizer.quant_model

    loss_fn = nn.CrossEntropyLoss().to(device)
    val_loader = load_data(train=False, data_dir=data_dir, batch_size=batch_size, subset_len=subset_len)

    # optional fast finetune
    if finetune and quant_mode == 'calib':
        ft_loader = load_data(train=False, data_dir=data_dir, batch_size=batch_size, subset_len=5120)
        quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
    elif finetune and quant_mode == 'test':
        quantizer.load_ft_param()

    acc1, loss_val = evaluate(quant_model, val_loader, loss_fn)

    print(f"Loss: {loss_val:.4f}")
    print(f"Top-1 accuracy: {acc1:.2f}%")

    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel(deploy_check=False)

# ======================================
# Main
# ======================================
if __name__ == '__main__':
    file_path = os.path.join(args.model_dir, "mnist_cnn.pth")

    print("-------- Start test: MNIST CNN --------")
    quantization(title="MNIST CNN quantization", file_path=file_path)
    print("-------- End of test --------")
