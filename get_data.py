import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Save the dataset to a CSV file in the data directory
df.to_csv('data/iris.csv', index=False)
dvc remote add -d my-mlops-storage gdrive://dvc-remote
print("Iris dataset saved to data/iris.csv")