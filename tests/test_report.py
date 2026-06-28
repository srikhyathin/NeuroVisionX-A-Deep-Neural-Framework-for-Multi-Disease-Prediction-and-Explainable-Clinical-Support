import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from backend.report_generator import ReportGenerator

report = ReportGenerator()

pdf = report.generate_report(
    patient_name="Sri Khyathi",
    age=20,
    gender="Female",
    prediction="Pneumonia",
    confidence=97.65,
    symptoms=[
        "Fever",
        "Cough",
        "Chest Pain"
    ],
    description="Pneumonia is a lung infection that causes inflammation in the air sacs.",
    precautions=[
        "Drink plenty of water",
        "Take medicines on time",
        "Get enough rest",
        "Consult a Pulmonologist"
    ],
    model_used="EfficientNet-B0"
)

print("Report Generated Successfully!")
print("Saved at:", pdf)