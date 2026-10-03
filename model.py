import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

# Load data
df = pd.read_csv('data/diabetes_prediction_dataset.csv')

# Encode categorical columns (gender, smoking_history)
le_gender = LabelEncoder()
le_smoking = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])
df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])

# Save encoders so app.py can use the SAME encoding later
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_smoking, 'le_smoking.pkl')

# Features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save metrics so the dashboard always shows the real, current numbers
report = classification_report(y_test, y_pred, output_dict=True)
metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(report["1"]["precision"], 4),
    "recall": round(report["1"]["recall"], 4),
    "f1": round(report["1"]["f1-score"], 4),
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save model (compressed to stay well under GitHub's file-size limits) and column order
joblib.dump(model, 'model.pkl', compress=3)
joblib.dump(list(X.columns), 'columns.pkl')
print("\nModel, encoders, and columns saved successfully.")
