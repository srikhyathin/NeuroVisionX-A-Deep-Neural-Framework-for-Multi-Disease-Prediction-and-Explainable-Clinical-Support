import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from backend.database import NeuroVisionDatabase

db = NeuroVisionDatabase()

db.save_prediction(

    patient_name="Sri Khyathi",

    age=20,

    gender="Female",

    disease="Pneumonia",

    confidence=98.54,

    model_used="EfficientNet",

    symptoms=[

        "Fever",

        "Chest Pain",

        "Cough"

    ],

    report_path="reports/test.pdf"

)

print(

    db.get_all_predictions()

)

print(

    "Total Predictions:",

    db.total_predictions()

)