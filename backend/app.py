from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import router

app = FastAPI(
    title="NeuroVisionX API",
    description="Multi-Modal Neural Intelligence for Early Disease Prediction",
    version="2.0.0"
)

# -----------------------------
# Enable CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Include Routes
# -----------------------------

app.include_router(router)

# -----------------------------
# Home Route
# -----------------------------

@app.get("/")
def home():

    return {
        "Project": "NeuroVisionX",
        "Version": "2.0",
        "Status": "Running",
        "AI Models": [
            "Symptoms Prediction",
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Brain Tumor Detection",
            "Pneumonia Detection"
        ]
    }

# -----------------------------
# Health Check
# -----------------------------

@app.get("/health")
def health():

    return {
        "status": "Healthy",
        "server": "Running"
    }

# -----------------------------
# API Information
# -----------------------------

@app.get("/about")
def about():

    return {

        "Name": "NeuroVisionX",

        "Description":
        "Unified Multi-Modal Neural Intelligence Platform for Early Disease Prediction",

        "Modules": [

            "Symptoms",

            "Diabetes",

            "Heart",

            "Brain Tumor",

            "Pneumonia",

            "Explainable AI",

            "NeuroFusion",

            "Medical Reports"

        ]
    }

# -----------------------------
# Run Server
# -----------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )