import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
from PIL import Image

# ------------------------
# Define Glaucoma UNet (exactly as provided)
# ------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        b = self.bottleneck(p3)
        u3 = self.up3(b)
        u3 = torch.cat([u3, d3], dim=1)
        c3 = self.conv3(u3)
        u2 = self.up2(c3)
        u2 = torch.cat([u2, d2], dim=1)
        c2 = self.conv2(u2)
        u1 = self.up1(c2)
        u1 = torch.cat([u1, d1], dim=1)
        c1 = self.conv1(u1)
        return self.out(c1)


# ------------------------
# Model Initialization
# ------------------------
MODEL_PATH = r"C:\Users\hitha\OneDrive\Desktop\Major project final\eyespy\dr_seg_roboflow\glucom_segmentation\unet_glaucoma_with_metrics.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

glaucoma_model = UNet(n_classes=3).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
# Strip "module." prefix if present
new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
glaucoma_model.load_state_dict(new_state)
glaucoma_model.eval()


# ------------------------
# Helper: CDR Computation
# ------------------------
def calculate_cdr(mask):
    cup = (mask == 1).astype(np.uint8)
    disc = (mask == 2).astype(np.uint8)
    cup_area = np.sum(cup)
    disc_area = np.sum(disc)
    area_cdr = cup_area / disc_area if disc_area > 0 else 0

    y_cup = np.any(cup, axis=1)
    y_disc = np.any(disc, axis=1)
    h_cup = np.sum(y_cup)
    h_disc = np.sum(y_disc)
    vertical_cdr = h_cup / h_disc if h_disc > 0 else 0

    cdr_value = max(area_cdr, vertical_cdr)
    return area_cdr, vertical_cdr, cdr_value


# ------------------------
# Inference Function
# ------------------------
def run_glaucoma_unet(image_path, save_folder, out_name_prefix="glaucoma_result"):
    """
    Runs Glaucoma U-Net on the given image.
    Returns (cdr_value, saved_image_path)
    """
    os.makedirs(save_folder, exist_ok=True)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image at {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Resize to model input size
    INPUT_SIZE = (256, 256)
    img_resized = cv2.resize(img_rgb, INPUT_SIZE)
    transform = transforms.ToTensor()
    inp = transform(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = glaucoma_model(inp)
        mask = torch.argmax(pred.squeeze(), dim=0).cpu().numpy().astype(np.uint8)

    # Compute CDR
    area_cdr, vert_cdr, cdr_value = calculate_cdr(mask)

    # Create color mask (cup=green, disc=red)
    mask_color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_color[mask == 1] = [0, 255, 0]   # cup
    mask_color[mask == 2] = [0, 0, 255]   # disc
    mask_color_resized = cv2.resize(mask_color, (orig_w, orig_h))

    # Overlay
    overlay = cv2.addWeighted(img_rgb, 0.7, mask_color_resized, 0.3, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # Save
    result_path = os.path.join(save_folder, f"{out_name_prefix}.png")
    cv2.imwrite(result_path, overlay_bgr)

    return cdr_value, os.path.basename(result_path)