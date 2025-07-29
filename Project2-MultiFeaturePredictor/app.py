from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            area = float(request.form['area'])
            bedrooms = float(request.form['bedrooms'])
            age = float(request.form['age'])
            prediction = model.predict([[area, bedrooms, age]])[0]
            prediction = round(prediction, 2)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
