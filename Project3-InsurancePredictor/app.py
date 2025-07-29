from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        prediction = model.predict([[age]])
        return render_template("index.html", prediction_text=f"Prediction: {'Will Buy' if prediction[0] == 1 else 'Will Not Buy'} Insurance")
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
