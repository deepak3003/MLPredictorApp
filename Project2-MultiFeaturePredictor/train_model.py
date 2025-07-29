import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Multi.csv")

# Assume these are your columns: area, bedrooms, age, price
X = df[['area', 'bedrooms', 'age']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved.")
