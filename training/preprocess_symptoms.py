import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------
# Load Original Dataset
# ---------------------------------------

df = pd.read_csv("data/symptoms/dataset.csv")

# Replace missing values
df.fillna("", inplace=True)

# ---------------------------------------
# Collect All Unique Symptoms
# ---------------------------------------

symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]

all_symptoms = set()

for col in symptom_columns:
    all_symptoms.update(
        s.strip()
        for s in df[col].astype(str)
        if s.strip() != ""
    )

all_symptoms = sorted(list(all_symptoms))

print("Total Unique Symptoms:", len(all_symptoms))

# ---------------------------------------
# Create Binary Feature Matrix
# ---------------------------------------

processed = pd.DataFrame(0, index=df.index, columns=all_symptoms)

for index, row in df.iterrows():

    for col in symptom_columns:

        symptom = str(row[col]).strip()

        if symptom != "":
            processed.at[index, symptom] = 1

# ---------------------------------------
# Encode Disease Labels
# ---------------------------------------

encoder = LabelEncoder()

processed["Disease"] = encoder.fit_transform(df["Disease"])

# ---------------------------------------
# Save Files
# ---------------------------------------

processed.to_csv(
    "data/symptoms/processed_symptoms.csv",
    index=False
)

joblib.dump(
    encoder,
    "saved_models/disease_encoder.pkl"
)

print("Preprocessing Completed.")
print("Number of Diseases:", len(encoder.classes_))
print("Number of Symptoms:", len(all_symptoms))