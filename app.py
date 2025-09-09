from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
model = joblib.load("donor_classifier_model.pkl")  # Load trained classifier

# ✅ Define safe diseases
safe_diseases = [
    "Anemia", "Asthma", "Bronchitis", "Dengue", "Diabetes", "Gastritis", "Gout",
    "Hypertension", "Hypothyroidism", "Hyperthyroidism", "Influenza", "Migraine", "Mumps",
    "Obesity", "Psoriasis", "Sinusitis", "Stomach Ulcer", "Tonsillitis", "Urinary Tract Infection (UTI)",
    "Vertigo", "Vitiligo", "Celiac Disease", "Endometriosis", "Rosacea", "No Disease"
]

@app.route("/category", methods=["POST"])
def category():
    data = request.get_json()

    try:
        age = int(data.get("age", 0))
        disease = str(data.get("healthStatus", "")).strip()
        donation_count = int(data.get("donationCount", 0))
        days_since_last_donation = int(data.get("daysSinceLastDonation", 0))

        # Check if disease is safe
        disease_safe = 1 if disease in safe_diseases else 0

        features = [age, disease_safe, donation_count, days_since_last_donation]
        prediction = model.predict([features])[0]

        return jsonify({
            "category": int(prediction),  # 0, 1, or 2
            "label": ["Less Suitable", "Suitable", "Highly Suitable"][int(prediction)]
        })

    except Exception as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400

# ✅ Use Render’s assigned port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default: 5000
    app.run(host="0.0.0.0", port=port, debug=False)