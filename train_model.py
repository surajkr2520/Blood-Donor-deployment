import pandas as pd
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
import joblib

# ✅ Define diseases that still allow blood donation
safe_diseases = [
    "Anemia", "Asthma", "Bronchitis", "Dengue", "Diabetes", "Gastritis", "Gout",
    "Hypertension", "Hypothyroidism", "Hyperthyroidism", "Influenza", "Migraine", "Mumps",
    "Obesity", "Psoriasis", "Sinusitis", "Stomach Ulcer", "Tonsillitis", "Urinary Tract Infection (UTI)",
    "Vertigo", "Vitiligo", "Celiac Disease", "Endometriosis", "Rosacea", "No Disease"
]

# 1. Load the dataset
df = pd.read_csv("donor_classification_dataset_v2.csv")

# 2. Encode disease: 1 if safe, else 0
df["disease_safe"] = df["Disease"].apply(lambda d: 1 if d in safe_diseases else 0)

# 3. Define features and label
X = df[["age", "disease_safe", "donation_count", "days_since_last_donation"]]
y = df["label"]

#model = RandomForestClassifier(
   # n_estimators=200,
    #max_depth=10,
    #max_features="sqrt",
   # min_samples_split=5,
    #random_state=42
#)
# 4. Train the Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=5, random_state=42)

model.fit(X, y)

# 5. Save the model
joblib.dump(model, "donor_classifier_model.pkl")
print("✅ Model trained and saved as donor_classifier_model.pkl")