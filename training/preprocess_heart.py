import pandas as pd

columns = [
    'age','sex','cp','trestbps',
    'chol','fbs','restecg',
    'thalach','exang','oldpeak',
    'slope','ca','thal','target'
]

df = pd.read_csv(
    "data/heart/processed.cleveland.data",
    names=columns
)

df.replace("?", pd.NA, inplace=True)

df.dropna(inplace=True)

df.to_csv(
    "data/heart/heart_processed.csv",
    index=False
)

print(df.shape)
print("Heart dataset processed.")