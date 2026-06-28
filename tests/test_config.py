import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from backend.config import *

print(BASE_DIR)
print(DATA_DIR)
print(MODEL_DIR)
print(REPORT_DIR)
print(DEVICE)
print(SYMPTOM_MODEL)