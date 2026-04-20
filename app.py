from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load models
diabetes_model = joblib.load("diabetes_model.pkl")
heart_model = joblib.load("heart_model.pkl")
ckd_model = joblib.load("ckd_model.pkl")




# Risk level
def risk_level(p):
    if p < 0.3:
        return "Low"
    elif p < 0.7:
        return "Medium"
    else:
        return "High"


# ---------------- ROUTE FOR FRONTEND ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- MAIN PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)   # ✅ FIX 1

    # Diabetes
    d_input = [[data["glucose"], data["bmi"], data["age"]]]
    d_prob = diabetes_model.predict_proba(d_input)[0][1]

    # Heart (✅ FIX 2: added exang)
    h_input = [[
        data["age"], data["sex"], data["cp"],
        data["trestbps"], data["chol"],
        data["thalch"], data["oldpeak"],
        data["exang"]
    ]]
    h_prob = heart_model.predict_proba(h_input)[0][1]

    # Kidney
    k_input = [[
        data["age"], data["bp"], data["bgr"],
        data["bu"], data["sc"], data["hemo"]
    ]]
    k_prob = ckd_model.predict_proba(k_input)[0][1]

    return jsonify({
        "diabetes": {"level": risk_level(d_prob), "score": d_prob},
        "heart": {"level": risk_level(h_prob), "score": h_prob},
        "ckd": {"level": risk_level(k_prob), "score": k_prob}
    })

if __name__ == "__main__":
    app.run(debug=True)