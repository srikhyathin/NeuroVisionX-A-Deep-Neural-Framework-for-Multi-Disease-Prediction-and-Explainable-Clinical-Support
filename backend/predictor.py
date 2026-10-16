import os
import joblib
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from models.symptom_model import SymptomNet
from models.diabetes_model import DiabetesNet
from models.heart_model import HeartNet
from models.image_model import BrainTumorNet

# =====================================================
# Project Paths
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_models"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

# =====================================================
# Device
# =====================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Running on {DEVICE}")

# =====================================================
# Load Encoders
# =====================================================

disease_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "disease_encoder.pkl"
    )
)

symptom_encoders = joblib.load(
    os.path.join(
        MODEL_DIR,
        "symptom_encoders.pkl"
    )
)

diabetes_scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "diabetes_scaler.pkl"
    )
)

# =====================================================
# Image Transform
# =====================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================================
# Load Symptom Dataset
# =====================================================

symptom_df = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "symptoms",
        "processed_symptoms.csv"
    )
)

NUM_SYMPTOMS = symptom_df.shape[1] - 1

print("Number of Symptoms:", NUM_SYMPTOMS)

# =====================================================
# Load Symptom Model
# =====================================================

symptom_model = SymptomNet(
    input_size=NUM_SYMPTOMS,
    num_classes=len(
        disease_encoder.classes_
    )
).to(DEVICE)

symptom_model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "symptom_model.pth"
        ),
        map_location=DEVICE
    )
)

symptom_model.eval()

# =====================================================
# Load Diabetes Model
# =====================================================

diabetes_model = DiabetesNet().to(
    DEVICE
)

diabetes_model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "diabetes_model.pth"
        ),
        map_location=DEVICE
    )
)

diabetes_model.eval()

# =====================================================
# Load Heart Model
# =====================================================

heart_model = HeartNet(
    input_size=13
).to(DEVICE)

heart_model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "heart_model.pth"
        ),
        map_location=DEVICE
    )
)

heart_model.eval()

# =====================================================
# Load Brain Tumor Model
# =====================================================

brain_model = BrainTumorNet(
    num_classes=4
).to(DEVICE)

brain_model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "brain_tumor_model.pth"
        ),
        map_location=DEVICE
    )
)

brain_model.eval()

# =====================================================
# Load Pneumonia Model
# =====================================================

pneumonia_model = BrainTumorNet(
    num_classes=2
).to(DEVICE)

pneumonia_model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "pneumonia_model.pth"
        ),
        map_location=DEVICE
    )
)

pneumonia_model.eval()

# =====================================================
# Class Labels
# =====================================================

brain_classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

pneumonia_classes = [
    "NORMAL",
    "PNEUMONIA"
]

# =====================================================
# Symptom Prediction
# =====================================================

def predict_symptoms(data):

    try:

        tensor = torch.tensor(
            [data],
            dtype=torch.float32
        ).to(DEVICE)

        with torch.no_grad():

            output = symptom_model(tensor)

            probability = F.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probability,
                dim=1
            )

        disease = disease_encoder.inverse_transform(
            [prediction.item()]
        )[0]

        return {

            "success": True,
            "prediction": disease,
            "confidence": round(
                confidence.item() * 100,
                2
            )

        }

    except Exception as e:

        return {

            "success": False,
            "error": str(e)

        }


# =====================================================
# Diabetes Prediction
# =====================================================

def predict_diabetes(data):

    try:

        scaled = diabetes_scaler.transform([data])

        tensor = torch.tensor(
            scaled,
            dtype=torch.float32
        ).to(DEVICE)

        with torch.no_grad():

            output = diabetes_model(tensor)

            probability = F.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probability,
                dim=1
            )

        result = (
            "Diabetic"
            if prediction.item() == 1
            else "Non-Diabetic"
        )

        return {

            "success": True,
            "prediction": result,
            "confidence": round(
                confidence.item() * 100,
                2
            )

        }

    except Exception as e:

        return {

            "success": False,
            "error": str(e)

        }


# =====================================================
# Heart Disease Prediction
# =====================================================

def predict_heart(data):

    try:

        tensor = torch.tensor(
            [data],
            dtype=torch.float32
        ).to(DEVICE)

        with torch.no_grad():

            output = heart_model(tensor)

            probability = F.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probability,
                dim=1
            )

        result = (
            "Heart Disease"
            if prediction.item() == 1
            else "Healthy"
        )

        return {

            "success": True,
            "prediction": result,
            "confidence": round(
                confidence.item() * 100,
                2
            )

        }

    except Exception as e:

        return {

            "success": False,
            "error": str(e)

        }


# =====================================================
# Brain Tumor Prediction
# =====================================================

def predict_brain(image_path):

    try:

        image = Image.open(image_path).convert("RGB")

        image = image_transform(image)

        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = brain_model(image)

            probability = F.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probability,
                dim=1
            )

        return {

            "success": True,
            "prediction": brain_classes[
                prediction.item()
            ],
            "confidence": round(
                confidence.item() * 100,
                2
            )

        }

    except Exception as e:

        return {

            "success": False,
            "error": str(e)

        }


# =====================================================
# Pneumonia Prediction
# =====================================================

def predict_pneumonia(image_path):

    try:

        image = Image.open(image_path).convert("RGB")

        image = image_transform(image)

        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = pneumonia_model(image)

            probability = F.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probability,
                dim=1
            )

        return {

            "success": True,
            "prediction": pneumonia_classes[
                prediction.item()
            ],
            "confidence": round(
                confidence.item() * 100,
                2
            )

        }

    except Exception as e:

        return {

            "success": False,
            "error": str(e)

        }