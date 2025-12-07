"""
Tri-intelligent — API FastAPI (version ResNet50)

- Pages:
    GET  /         -> index.html (présentation)
    GET  /webcam   -> webcam.html (capture et classification)
- API:
    POST /predict  -> JSON {label, proba, bin, bin_color}

Dépendances: fastapi, uvicorn, jinja2, python-multipart, torch, torchvision, pillow
Arborescence attendue:
project/
  app/
    main.py                 <-- ce fichier
    templates/
      index.html
      webcam.html
    static/
      style.css
  checkpoints/
    class_mapping.json
    model_resnet50.pth
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

# ---------- Config chemins ----------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
CKPT_DIR = BASE_DIR.parent / "checkpoints" if (BASE_DIR / "templates").exists() else BASE_DIR.parent / "checkpoints"

# ---------- Prétraitements ----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------- Chargement mapping ----------
mapping_path = CKPT_DIR / "class_mapping.json"
if not mapping_path.exists():
    raise RuntimeError(f"Fichier de mapping manquant: {mapping_path}.")
with mapping_path.open("r", encoding="utf-8") as f:
    idx2class: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}

# ---------- Chargement modèle ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(idx2class))

state_path = CKPT_DIR / "model_resnet50.pth"
if not state_path.exists():
    raise RuntimeError(f"Fichier de poids manquant: {state_path}.")
state = torch.load(state_path, map_location=device)
model.load_state_dict(state, strict=True)
model.eval().to(device)

# ---------- Règles de bacs ----------
BIN_RULES = {
    # Mots-clés -> Bac
    "glass": "Verre",
    "verre": "Verre",

    "plastic": "Plastique",
    "plastique": "Plastique",

    "metal": "Métal",
    "can": "Métal",
    "aluminum": "Métal",

    "paper": "Emballages & papiers",
    "cardboard": "Emballages & papiers",
    "papier": "Emballages & papiers",
    "emballage": "Emballages & papiers",

    # Autres classes fréquentes du dataset v2
    "battery": "Non recyclable",
    "trash": "Non recyclable",
    "clothes": "Non recyclable",
    "shoes": "Non recyclable",
    "biological": "Non recyclable",
}
BIN_COLORS = {
    "Verre": "#2ecc71",
    "Plastique": "#f39c12",
    "Métal": "#bdc3c7",
    "Emballages & papiers": "#3498db",
    "Non recyclable": "#7f8c8d",
}

def to_bin(label: str) -> str:
    l = label.lower()
    for key, bin_name in BIN_RULES.items():
        if key in l:
            return bin_name
    return "Non recyclable"

# ---------- App FastAPI ----------
app = FastAPI(title="Tri-intelligent API (ResNet50)")

# Fichiers statiques & templates
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR if TEMPLATES_DIR.exists() else BASE_DIR)

# ---------- Routes pages ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Page d'accueil."""
    # cherche index.html dans templates, sinon à la racine du dossier courant
    template_name = "index.html" if (TEMPLATES_DIR / "index.html").exists() else "index.html"
    return templates.TemplateResponse(template_name, {"request": request})

@app.get("/webcam", response_class=HTMLResponse)
async def webcam_page(request: Request) -> HTMLResponse:
    """Page webcam."""
    template_name = "webcam.html" if (TEMPLATES_DIR / "webcam.html").exists() else "webcam.html"
    return templates.TemplateResponse(template_name, {"request": request})

# ---------- API prédiction ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, str | float]:
    """Prend une image et renvoie la prédiction + bac."""
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).max().item()
        idx = int(logits.argmax(1).item())

    label = idx2class[idx]
    bin_name = to_bin(label)

    return {
        "label": label,
        "proba": round(prob, 4),
        "bin": bin_name,
        "bin_color": BIN_COLORS.get(bin_name, "#7f8c8d"),
    }

# ---------- Lancement local ----------
# python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
