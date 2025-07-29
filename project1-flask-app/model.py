# save_model.py (run this in the same venv you're using now)

import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Dummy data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)

# Save the model safely
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
