import pandas as pd

symptoms = pd.read_csv("data/symptoms/dataset.csv")
print("Symptoms Shape:", symptoms.shape)
print("Diseases:", symptoms["Disease"].nunique())

diabetes = pd.read_csv("data/diabetes/diabetes.csv")
print("Diabetes Shape:", diabetes.shape)