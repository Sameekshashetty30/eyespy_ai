# backend/utils/dr_inference.py
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# ------------------------
# UNet implementation (exact as provided)
# ------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(self.maxpool(conv1))
        conv3 = self.dconv_down3(self.maxpool(conv2))
        conv4 = self.dconv_down4(self.maxpool(conv3))

        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        x = self.conv_last(x)
        return torch.sigmoid(x)


# ------------------------
# Model init & load
# ------------------------
MODEL_PATH = r"C:\Users\hitha\OneDrive\Desktop\eye disease app\backend\models\unet_dr_best_collab.pt"
DEVICE = torch.device("cpu")

def _load_model(model, path):
    """Load .pt file whether it's a state_dict, checkpoint dict, or full model object."""
    loaded = torch.load(path, map_location=DEVICE)

    if isinstance(loaded, dict) and not isinstance(loaded, OrderedDict):
        if 'state_dict' in loaded:
            state = loaded['state_dict']
        elif 'model_state' in loaded:
            state = loaded['model_state']
        else:
            state = loaded

        try:
            model.load_state_dict(state)
        except RuntimeError:
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(new_state)
        return model

    if isinstance(loaded, OrderedDict) or (
        isinstance(loaded, dict) and all(isinstance(v, torch.Tensor) for v in loaded.values())
    ):
        try:
            model.load_state_dict(loaded)
        except RuntimeError:
            new_state = {k.replace("module.", ""): v for k, v in loaded.items()}
            model.load_state_dict(new_state)
        return model

    if hasattr(loaded, "eval"):
        return loaded.to(DEVICE)

    raise ValueError("Invalid model format")


# Load UNet
unet_model = UNet(in_ch=3, out_ch=1)

try:
    unet_model = _load_model(unet_model, MODEL_PATH)
    unet_model.to(DEVICE)
    unet_model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load DR UNet model from {MODEL_PATH}: {e}")


# ------------------------
# Inference function (UPDATED with bbox_count)
# ------------------------
def run_dr_unet(image_path, save_folder, out_name_prefix="dr_result"):
    """
    Runs DR U-Net, draws bounding boxes, and returns:
        (result_filename, bbox_count)
    """
    os.makedirs(save_folder, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    INPUT_SIZE = (256, 256)
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = unet_model(tensor)
        out_np = out.squeeze().cpu().numpy()

    mask = (out_np > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    orig_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = orig_bgr.copy()
    bbox_count = 0   # <=== added

    # Draw bounding boxes
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if w > 15 and h > 15 and area > 40:
            bbox_count += 1  # <=== count boxes

            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(overlay, "Lesion", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    blended = cv2.addWeighted(orig_bgr, 0.8, overlay, 0.4, 0)

    out_filename = f"{out_name_prefix}_bbox.png"
    result_path = os.path.join(save_folder, out_filename)
    cv2.imwrite(result_path, blended)

    # Return filename + bbox count
    return os.path.basename(result_path), bbox_count
