from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

species = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[key]) for key in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        prediction = model.predict([features])[0]
        predicted_species = species[prediction]
        return render_template('index.html', prediction_text=f'Predicted Iris Species: {predicted_species}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
