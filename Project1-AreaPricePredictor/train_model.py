# train_model.ipynb or train_model.py

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("areaprice071.csv")


# Features and label
X = df[['area']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
