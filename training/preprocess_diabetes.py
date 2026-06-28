import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================================
# Create folders if they don't exist
# =====================================================

os.makedirs("saved_models", exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("data/diabetes/diabetes.csv")

print("=" * 50)
print("NeuroVisionX - Diabetes Preprocessing")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# Split Features and Target
# =====================================================

X = df.drop("Outcome", axis=1)

y = df["Outcome"]

# =====================================================
# Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# Feature Scaling
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# =====================================================
# Save Scaler
# =====================================================

joblib.dump(
    scaler,
    "saved_models/diabetes_scaler.pkl"
)

# =====================================================
# Save Processed Dataset
# =====================================================

train_df = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

train_df["Outcome"] = y_train.values

test_df = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)

test_df["Outcome"] = y_test.values

train_df.to_csv(
    "data/diabetes/diabetes_train.csv",
    index=False
)

test_df.to_csv(
    "data/diabetes/diabetes_test.csv",
    index=False
)

# =====================================================
# Display Information
# =====================================================

print("\nTraining Shape :", train_df.shape)
print("Testing Shape  :", test_df.shape)

print("\nScaler Saved Successfully!")

print("\nProcessed datasets saved.")

print("=" * 50)