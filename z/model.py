import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# ✅ Sample training data (realistic prices)
data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'price': [14000, 21000, 28000, 35000, 42000]
}
df = pd.DataFrame(data)

# Train the model
model = LinearRegression()
model.fit(df[['area']], df['price'])

# Save the model
joblib.dump(model, 'model.pkl')

print("✅ Model trained and saved as model.pkl")
