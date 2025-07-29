from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load model
model_path = "model.pkl"
if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))
else:
    model = None
    print("❌ model.pkl not found!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return "Model not loaded. Please upload model.pkl."

    # Example: Get input from form (adjust input keys based on your form)
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]

    return render_template("index.html", prediction_text=f"Predicted result: {prediction}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
