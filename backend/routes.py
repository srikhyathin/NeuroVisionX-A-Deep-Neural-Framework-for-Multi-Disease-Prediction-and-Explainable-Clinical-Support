from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "Welcome to NeuroVisionX API"
    }


@router.get("/status")
def status():
    return {
        "status": "Running Successfully"
    }


@router.get("/about")
def about():
    return {
        "Project": "NeuroVisionX",
        "Version": "2.0",
        "Developer": "Sri Khyathi"
    }