# NeuroVisionX: A Deep Neural Framework for Multi-Disease Prediction and Explainable Clinical Support

**NeuroVisionX** is an AI-powered healthcare support platform that integrates multiple independently trained deep-learning and neural-network models for preliminary disease prediction. It combines symptom-based classification, clinical data analysis, medical image classification, explainable AI, patient history management, and automated PDF report generation through a unified Streamlit interface.

> **Disclaimer:** NeuroVisionX is an academic prototype intended for research and educational purposes. Its predictions are not a substitute for professional medical diagnosis or treatment.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Techniques Used](#techniques-used)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Model Evaluation](#model-evaluation)
- [Explainability](#explainability)
- [Patient History and Reports](#patient-history-and-reports)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Overview

Healthcare data comes in different forms, including patient-reported symptoms, structured clinical parameters, and medical images. Analyzing these data types often requires separate diagnostic tools.

NeuroVisionX provides a centralized application containing five independent prediction modules:

1. Symptom-based disease prediction across 41 disease classes.
2. Heart disease prediction using clinical parameters.
3. Diabetes prediction using patient health attributes.
4. Brain tumor classification using MRI images.
5. Pneumonia detection using chest X-ray images.

The platform also supports image explainability, patient diagnostic history, analytics, and downloadable PDF reports.

**Important:** The five prediction modules are independently trained. NeuroVisionX integrates them into a common application; it does not currently perform joint multimodal feature fusion.

## Problem Statement

- Separate disease prediction systems can make healthcare data analysis less centralized.
- Manual interpretation of medical images can be time-consuming.
- Prediction results may be difficult to interpret without visual explanations.
- Patient diagnostic records and reports can be challenging to manage across separate tools.
- An integrated platform is needed to organize preliminary predictions, explanations, patient history, and reporting.

## Objectives

- Develop an AI-powered platform for preliminary disease prediction.
- Implement symptom-based, clinical-data, and medical image prediction modules.
- Apply explainable AI to improve the interpretability of image predictions.
- Maintain patient diagnostic history using a database.
- Generate downloadable PDF reports.
- Provide a user-friendly Streamlit interface for accessing the modules.

## Key Features

### 1. Symptom-Based Disease Prediction
- Accepts user-reported symptoms.
- Uses symptom encoding and a neural-network classifier.
- Supports prediction across 41 disease classes.

### 2. Heart Disease Prediction
- Accepts structured clinical parameters.
- Uses a trained neural-network model to generate a prediction.

### 3. Diabetes Prediction
- Analyzes patient health attributes.
- Uses preprocessing and a trained neural-network model.

### 4. Brain Tumor Classification
- Processes brain MRI images.
- Uses an EfficientNet-B0-based CNN model.
- Supports Grad-CAM visualization where implemented.

### 5. Pneumonia Detection
- Processes chest X-ray images.
- Uses an EfficientNet-B0-based CNN model.
- Supports visual explanation through Grad-CAM where implemented.

### 6. Explainable AI
- **Grad-CAM:** Highlights image regions that influence image-model predictions.
- **SHAP:** Explains feature contributions for supported prediction models.

### 7. Patient History and Reporting
- Stores and retrieves diagnostic records using SQLite.
- Generates downloadable PDF reports.
- Provides access to prediction history and analytics.

## System Architecture

The application follows a modular architecture:

1. **User Interface:** Streamlit pages for entering symptoms, clinical values, and medical images.
2. **Input Preprocessing:** Data cleaning, encoding, normalization, and image resizing.
3. **Prediction Modules:** Five independently trained disease prediction models.
4. **Explainability:** Grad-CAM and SHAP for supported models.
5. **Data Management:** SQLite database for patient history.
6. **Report Generation:** PDF creation using the report-generation module.
7. **Results Interface:** Displays predictions, supported explanations, history, and reports.

## Techniques Used

| Technique | Application |
|---|---|
| Artificial Neural Networks (ANN) | Heart disease and diabetes prediction |
| Neural-network classification | Symptom-based prediction |
| CNN and transfer learning | Brain tumor and pneumonia image classification |
| Image preprocessing | Resizing and normalization |
| Grad-CAM and SHAP | Model explainability |
| Model evaluation | Accuracy, precision, recall, and F1-score |
| SQLite and PDF generation | Patient history and reporting |

## Technology Stack

| Component | Technologies |
|---|---|
| Programming language | Python |
| Frontend | Streamlit |
| Machine learning | Scikit-learn |
| Deep learning | PyTorch and the framework used by each trained model |
| Image models | EfficientNet-B0 |
| Explainable AI | Grad-CAM, SHAP |
| Database | SQLite |
| Report generation | ReportLab |
| Data processing | Pandas, NumPy |
| Version control | Git and GitHub |

*The exact libraries and versions should match the project's `requirements.txt` and implementation.*

## Project Structure

The following is a representative structure. Keep the actual filenames and directories in your repository if they differ.

```text
NeuroVisionX/
├── backend/
│   ├── predictor.py
│   ├── predict.py
│   └── report_generator.py
├── frontend/
│   ├── Home.py
│   ├── pages/
│   │   ├── Symptoms.py
│   │   ├── PatientHistory.py
│   │   ├── Heart.py
│   │   ├── Diabetes.py
│   │   ├── BrainTumor.py
│   │   ├── Pneumonia.py
│   │   ├── Analytics.py
│   │   ├── Explainability.py
│   │   └── About.py
│   └── neurovisionx.db
├── data/
│   └── symptoms/
├── models/
│   ├── disease_encoder.pkl
│   ├── symptom_encoders.pkl
│   ├── symptom_model.pth
│   ├── diabetes_model.pth
│   ├── heart_model.pth
│   ├── brain_tumor_model.pth
│   └── pneumonia_model.pth
├── training/
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** Model files may be stored in `saved_models/` or other locations in your implementation. Update this tree to match the actual repository.

## Installation and Setup

### Prerequisites

Install the following:

- Python compatible with your project's dependencies.
- Visual Studio Code.
- Git.
- The trained model files required by the application.

### Step 1: Clone the Repository

```bash
git clone https://github.com/srikhyathin/NeuroVisionX-A-Deep-Neural-Framework-for-Multi-Disease-Prediction-and-Explainable-Clinical-Support.git
```

Move into the project directory:

```bash
cd NeuroVisionX-A-Deep-Neural-Framework-for-Multi-Disease-Prediction-and-Explainable-Clinical-Support
```

### Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If a dependency installation fails, check the Python version and the package versions required by your trained models.

### Step 4: Verify Model Files

Confirm that the required `.pth` and `.pkl` files exist in the directories expected by the model-loading code.

If the trained models are not included in GitHub, obtain them from your authorized project storage and place them in the expected locations.

## Running the Application

Open the project folder in VS Code and activate the virtual environment.

Identify the Streamlit entry-point file from your project. If the entry point is `frontend/Home.py`, run:

```bash
python -m streamlit run frontend/Home.py
```

If your application uses a different entry point, substitute its actual path.

Streamlit normally displays a local URL such as:

```text
http://localhost:8501
```

Open the URL in your browser.

## Model Evaluation

The prediction models should be evaluated independently using held-out test data.

Recommended evaluation metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted positive cases that are correct.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

For multiclass symptom prediction, include class-wise performance and a confusion matrix where available.

For brain tumor and pneumonia classification, report test-set results for the corresponding image datasets.

**Results must be taken from actual evaluation outputs.** Do not treat training accuracy, prediction confidence, or illustrative figures as test accuracy. The final reported metrics should match the evaluated test set.

## Explainability

### Grad-CAM
Grad-CAM generates a heatmap that highlights image regions contributing to a CNN's prediction. In NeuroVisionX, it is intended to help users inspect brain MRI and chest X-ray model outputs.

### SHAP
SHAP estimates feature contributions to model predictions for supported modules. Its availability and explanation method depend on the model implementation.

Explanations indicate model behavior and should not be interpreted as proof that a highlighted region or feature caused a disease.

## Patient History and Reports

### SQLite Patient History
The application uses SQLite to manage diagnostic records, including supported patient details, inputs, predictions, and timestamps.

### PDF Report Generation
The report-generation module produces downloadable PDF reports containing the relevant prediction details and other supported information.

**Privacy:** Do not upload real patient records, personally identifiable health information, database files containing private records, or credentials to a public repository.

## Limitations

- The models are trained independently rather than through joint multimodal fusion.
- Prediction quality depends on the training data, preprocessing, and evaluation methodology.
- Performance may vary across patient populations, imaging devices, and clinical settings.
- Model confidence is not necessarily a calibrated measure of uncertainty.
- Grad-CAM and SHAP explanations do not establish clinical correctness.
- The application requires further independent validation before any clinical use.

## Future Enhancements

- Evaluate models on independent and more diverse datasets.
- Investigate calibrated uncertainty estimation.
- Explore validated multimodal fusion across clinical data and medical images.
- Improve model monitoring and error analysis.
- Strengthen privacy, access control, and secure data handling.
- Add comprehensive automated tests and reproducible evaluation reports.

## Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Commit your changes with a descriptive message.
5. Submit a pull request.

Please do not commit secrets, private patient information, or unlicensed datasets.

## License

A license has not been specified here. Add an appropriate `LICENSE` file before permitting others to reuse, modify, or redistribute the project.

## Author

**Sri Khyathi Nagendla**

GitHub: [@srikhyathin](https://github.com/srikhyathin)

---

*NeuroVisionX — An academic AI platform for modular disease prediction, explainability, and clinical-support workflows.*
