# modeling.py — Final Fix with Verified Model Saving

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ✅ Load dataset without headers
data_path = "C:/Gokul Important things/Content Monetization Modeler/outputs/cleaned/cleaned_dataset.csv"
df = pd.read_csv(data_path, header=None)

# ✅ Assign correct column names manually
df.columns = [
    'views', 'likes', 'comments', 'watch_time_minutes',
    'video_length_minutes', 'subscribers', 'title', 'tags',
    'category', 'upload_day', 'device', 'country', 'language',
    'channel_type', 'engagement_score', 'ctr', 'cpm', 'rpm',
    'retention_rate', 'log_revenue'
]

# ✅ Define input features and target
feature_columns = [
    'views', 'likes', 'comments', 'watch_time_minutes',
    'video_length_minutes', 'subscribers'
]
target_column = 'log_revenue'

# ✅ Convert numeric columns and fill missing values
for col in feature_columns + [target_column]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# ✅ Fill missing values in non-numeric columns
for col in df.columns:
    if col not in feature_columns + [target_column] and df[col].dtype == 'object':
        df[col] = df[col].fillna("unknown")

# ✅ Prepare training data
X = df[feature_columns].copy()
y = df[target_column].copy()

print("✅ Training on features:", X.columns.tolist())

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# ✅ Evaluate models
best_model = None
best_score = -np.inf

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name} → R²: {r2:.4f}, RMSE: {rmse:.4f}")
    
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

# ✅ Save model with feature metadata
model_bundle = {
    "model": best_model,
    "features": feature_columns
}

# ✅ Ensure absolute path and folder creation
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "..", "outputs", "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, f"{best_name}_model.pkl")
joblib.dump(model_bundle, model_path)

print(f"\n✅ Best model saved: {best_name} with R² = {best_score:.4f}")
print(f"📦 Model saved to: {model_path}")
