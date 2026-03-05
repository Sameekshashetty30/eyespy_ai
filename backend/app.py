# backend/app.py — FINAL UPGRADED VERSION (error-free)

import os
import warnings
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request, redirect
from PIL import Image
import numpy as np
import cv2
import pywt

warnings.filterwarnings("ignore", category=FutureWarning)

from backend.utils.dr_inference import run_dr_unet
from backend.utils.glaucoma_inference import run_glaucoma_unet


# -------------------------
# ResNet50 — 6-channel classifier
# -------------------------
class ResNet50_6ch(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        try:
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
        except:
            resnet = models.resnet50(pretrained=pretrained)

        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            resnet.conv1.weight[:, :3] = old_conv.weight
            mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
            resnet.conv1.weight[:, 3:] = mean_rgb

        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


# -------------------------
# Flask setup
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

UPLOAD_FOLDER = os.path.join(STATIC_DIR, "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Load disease classifier
# -------------------------
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
CLASS_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_wavelet_resnet50.pt")

classifier = ResNet50_6ch(num_classes=len(CLASS_NAMES))
checkpoint = torch.load(CLASS_MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and ("model" in checkpoint or "state_dict" in checkpoint):
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
else:
    state_dict = checkpoint

fixed_state = {}
for k, v in state_dict.items():
    k = k.replace("module.", "").replace("model.", "")
    if not k.startswith("resnet."):
        k = f"resnet.{k}"
    fixed_state[k] = v

classifier.load_state_dict(fixed_state, strict=False)
classifier = classifier.to(DEVICE)
classifier.eval()


# -------------------------
# Lightweight Fundus Detector
# -------------------------
class FundusDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)


fundus_detector = FundusDetector().to(DEVICE)
FUNDUS_MODEL_PATH = os.path.join(BASE_DIR, "models", "fundus_vs_nonfundus.pt")

if os.path.exists(FUNDUS_MODEL_PATH):
    fd_state = torch.load(FUNDUS_MODEL_PATH, map_location=DEVICE)
    new_state = {}

    for k, v in fd_state.items():
        if k.startswith("features") or k.startswith("classifier"):
            new_state[f"model.{k}"] = v
        else:
            new_state[k] = v

    fundus_detector.load_state_dict(new_state, strict=False)
    print("✅ Fundus Detector loaded successfully.")
else:
    print("⚠️ fundus_vs_nonfundus.pt NOT FOUND — fallback enabled!")

fundus_detector.eval()


# -------------------------
# Preprocessing helper
# -------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_wavelet(img_pil, wavelet="db2", size=(224, 224)):
    img = img_pil.convert("RGB").resize(size)
    img_np = np.array(img).astype(np.float32) / 255.0

    green = img_np[:, :, 1]
    cA, (cH, cV, cD) = pywt.dwt2(green, wavelet)

    up = lambda x: cv2.resize(x, (img_np.shape[1], img_np.shape[0]))
    wave = np.stack([up(cH), up(cV), up(cD)], axis=-1)
    wave = (wave - wave.min()) / (wave.max() - wave.min() + 1e-8)

    rgb_t = torch.tensor(img_np).permute(2, 0, 1)
    wave_t = torch.tensor(wave).permute(2, 0, 1)
    x = torch.cat([rgb_t, wave_t], dim=0)

    x[:3] = (x[:3] - IMAGENET_MEAN) / IMAGENET_STD
    return x.unsqueeze(0)


# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # FILE LOAD
    if "image" not in request.files:
        return redirect(request.url)

    file = request.files["image"]
    if file.filename == "":
        return redirect(request.url)

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    img_pil = Image.open(filepath).convert("RGB")

    # -------------------------------
    # STEP 1 — FUNDUS DETECTION
    # -------------------------------
    preprocess_fd = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    img_fd = preprocess_fd(img_pil).unsqueeze(0).to(DEVICE)
    is_fundus = True

    if os.path.exists(FUNDUS_MODEL_PATH):
        with torch.no_grad():
            pred_fd = fundus_detector(img_fd)
            probs_fd = torch.softmax(pred_fd, 1)[0]

            nonfundus_prob = probs_fd[0].item()
            fundus_prob = probs_fd[1].item()

            print("\n========== FUNDUS-DETECTOR ==========")
            print(f"Non-Fundus Probability : {nonfundus_prob:.4f}")
            print(f"Fundus Probability     : {fundus_prob:.4f}")
            print("Decision:", "FUNDUS ✔" if fundus_prob > nonfundus_prob else "NON-FUNDUS ✘")
            print("=====================================\n")

            if nonfundus_prob > 0.60:
                is_fundus = False

    if not is_fundus:
        return render_template(
            "result.html",
            disease="Invalid Image",
            error_message="⚠️ Please upload a valid fundus image.",
            original_image=os.path.basename(filepath),
            result_image=None,
            bbox_count=0
        )

    # -------------------------------
    # STEP 2 — DISEASE CLASSIFICATION
    # -------------------------------
    input_tensor = preprocess_wavelet(img_pil).to(DEVICE)

    with torch.no_grad():
        out = classifier(input_tensor)
        probs = torch.softmax(out, 1)[0].cpu().numpy()
        pred = torch.argmax(out, 1).item()
        disease = CLASS_NAMES[pred]

        print("\n========== DISEASE CLASSIFIER ==========")
        for cname, pval in zip(CLASS_NAMES, probs):
            print(f"{cname:<25} : {pval:.4f}")
        print("Predicted:", disease)
        print("========================================\n")

    original_name = os.path.basename(filepath)

    # -------------------------------
    # STEP 3 — CONDITIONAL INFERENCE
    # -------------------------------
    if disease == "Diabetic Retinopathy":
        result_filename, bbox_count = run_dr_unet(filepath, UPLOAD_FOLDER)

        return render_template(
            "result.html",
            disease=disease,
            bbox_count=bbox_count,
            original_image=original_name,
            result_image=result_filename
        )

    elif disease == "Glaucoma":
        cdr_value, result_filename = run_glaucoma_unet(filepath, UPLOAD_FOLDER)

        return render_template(
            "result.html",
            disease=disease,
            cdr=f"{cdr_value:.2f}",
            original_image=original_name,
            bbox_count=0,
            result_image=result_filename
        )

    else:
        return render_template(
            "result.html",
            disease=disease,
            original_image=original_name,
            bbox_count=0,
            result_image=None
        )


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
