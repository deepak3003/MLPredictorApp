import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("insurance_data.csv")

# Features and target
X = df[['age']]  # Only 'age' column is available
y = df['bought_insurance']

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
