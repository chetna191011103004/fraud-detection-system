# fraud_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load CSV
df = pd.read_csv("transactions.csv")

# Features aur target split karo
X = df[["amount","location","device","transaction_type"]]
y = df["is_fraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model train karo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model save karo
joblib.dump(model, "fraud_model.pkl")

print("Model trained and saved successfully!")