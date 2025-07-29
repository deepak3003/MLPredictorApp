import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample training data
data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'price': [14400, 1570, 2700, 2507, 3700]
}
df = pd.DataFrame(data)

# Train model
model = LinearRegression()
model.fit(df[['area']], df['price'])

# Save the model
joblib.dump(model, 'model.pkl')

print("✅ Model trained and saved as model.pkl")
