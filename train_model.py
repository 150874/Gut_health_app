import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

print("Loading optimized dataset...")
# 1. Load the data
df = pd.read_csv('optimized_meal_logs_dataset.csv')

# 2. Select Features (X) and Target (y)
# We drop User_ID and Timestamp because the model shouldn't learn from names/dates.
# We also drop the Target variable from X.
X_raw = df.drop(columns=['User_ID', 'Timestamp', 'Symptom_Flare_Up_Score'])
y = df['Symptom_Flare_Up_Score']

# 3. Handle Categorical Data (One-Hot Encoding)
# ML models only understand numbers. This converts text like "GERD" into 1s and 0s.
X = pd.get_dummies(X_raw)

# Save the exact column names so your Flask app knows how to format user input later
expected_columns = X.columns.tolist()
joblib.dump(expected_columns, 'model_columns.pkl')

# 4. Split the data into Training and Testing sets
# We use 80% of data to teach the model, and keep 20% hidden to test it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model (The "Brain")
print("Training the Random Forest model (this might take a few seconds)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Test the Model's Accuracy
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print(f"Model successfully trained! Average Prediction Error: +/- {error:.2f} symptom points.")

# 7. Save the trained model to a file
joblib.dump(model, 'gut_health_model.pkl')
print("Model saved as 'gut_health_model.pkl'. Ready for Flask!")