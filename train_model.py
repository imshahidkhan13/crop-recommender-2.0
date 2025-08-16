import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# --- Feature for "Reverse Lookup" with MIN/MAX ranges ---
# Group by crop and aggregate to find the min and max for each nutrient
crop_ranges = df.groupby('label').agg({
    'N': ['min', 'max'],
    'P': ['min', 'max'],
    'K': ['min', 'max'],
    'temperature': ['min', 'max'],
    'humidity': ['min', 'max'],
    'ph': ['min', 'max'],
    'rainfall': ['min', 'max']
})
# Flatten the multi-level column names (e.g., from ('N', 'min') to 'N_min')
crop_ranges.columns = ['_'.join(col).strip() for col in crop_ranges.columns.values]
crop_ranges.to_csv('crop_ranges.csv') # Save to a new file
print("Saved crop ranges data to crop_ranges.csv")


# --- Model Training (no changes here) ---
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'crop_model.joblib')
print("Model trained and saved as crop_model.joblib")