import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

print("Loading dataset...")
# Update dataset path if card_transdata.csv is in another folder
df = pd.read_csv("card_transdata.csv") 

# Prepare features
X = df.drop("fraud", axis=1)

print("Scaling features...")
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

print("Training Isolation Forest model...")
Is = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
Is.fit(X_scaled)

print("Saving model and scaler...")
joblib.dump(sc, 'scaler.pkl')
joblib.dump(Is, 'Isolation_forest_model.pkl')

print("Successfully generated scaler.pkl and Isolation_forest_model.pkl!")