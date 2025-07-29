from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    prediction = model.predict([[area]])
    return render_template('index.html', prediction=round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
