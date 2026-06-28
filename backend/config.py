import os
import torch

# ==========================================================
# Project Directories
# ==========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

REPORT_DIR = os.path.join(BASE_DIR, "reports")

DATABASE_PATH = os.path.join(BASE_DIR, "neurovisionx.db")

# Create reports folder if not exists
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================================
# Device Configuration
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Model Paths
# ==========================================================

SYMPTOM_MODEL = os.path.join(
    MODEL_DIR,
    "symptom_model.pth"
)

DIABETES_MODEL = os.path.join(
    MODEL_DIR,
    "diabetes_model.pth"
)

HEART_MODEL = os.path.join(
    MODEL_DIR,
    "heart_model.pth"
)

BRAIN_MODEL = os.path.join(
    MODEL_DIR,
    "brain_tumor_model.pth"
)

PNEUMONIA_MODEL = os.path.join(
    MODEL_DIR,
    "pneumonia_model.pth"
)

NEUROFUSION_MODEL = os.path.join(
    MODEL_DIR,
    "neurofusion.pth"
)

# ==========================================================
# Encoder Files
# ==========================================================

DISEASE_ENCODER = os.path.join(
    MODEL_DIR,
    "disease_encoder.pkl"
)

SYMPTOM_ENCODERS = os.path.join(
    MODEL_DIR,
    "symptom_encoders.pkl"
)

# ==========================================================
# Dataset Paths
# ==========================================================

SYMPTOM_DATASET = os.path.join(
    DATA_DIR,
    "symptoms",
    "processed_symptoms.csv"
)

DIABETES_DATASET = os.path.join(
    DATA_DIR,
    "diabetes",
    "diabetes.csv"
)

HEART_DATASET = os.path.join(
    DATA_DIR,
    "heart",
    "heart_processed.csv"
)

BRAIN_DATASET = os.path.join(
    DATA_DIR,
    "brain_tumor"
)

PNEUMONIA_DATASET = os.path.join(
    DATA_DIR,
    "pneumonia"
)

DESCRIPTION_FILE = os.path.join(
    DATA_DIR,
    "symptoms",
    "symptom_Description.csv"
)

PRECAUTION_FILE = os.path.join(
    DATA_DIR,
    "symptoms",
    "symptom_precaution.csv"
)

# ==========================================================
# Image Settings
# ==========================================================

IMAGE_SIZE = (224, 224)

BRAIN_CLASSES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

PNEUMONIA_CLASSES = [
    "NORMAL",
    "PNEUMONIA"
]

# ==========================================================
# API Settings
# ==========================================================

API_TITLE = "NeuroVisionX API"

API_VERSION = "2.0.0"

API_DESCRIPTION = (
    "Multi-Modal Neural Intelligence for Early Disease Prediction"
)

# ==========================================================
# Report Settings
# ==========================================================

REPORT_AUTHOR = "NeuroVisionX"

REPORT_HOSPITAL = "NeuroVisionX AI Healthcare"

# ==========================================================
# Database
# ==========================================================

DATABASE_NAME = "neurovisionx.db"