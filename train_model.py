import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
	mean_absolute_error,
	mean_squared_error,
	r2_score,
	confusion_matrix,
	recall_score,
	precision_score,
	f1_score,
	balanced_accuracy_score,
	roc_auc_score,
	make_scorer,
)
import joblib
import json

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


def to_binary(values, threshold):
	return (values >= threshold).astype(int)


def recall_flare(y_true, y_pred, threshold):
	return recall_score(to_binary(y_true, threshold), to_binary(y_pred, threshold), zero_division=0)


def precision_flare(y_true, y_pred, threshold):
	return precision_score(to_binary(y_true, threshold), to_binary(y_pred, threshold), zero_division=0)


def f1_flare(y_true, y_pred, threshold):
	return f1_score(to_binary(y_true, threshold), to_binary(y_pred, threshold), zero_division=0)


def balanced_accuracy_flare(y_true, y_pred, threshold):
	return balanced_accuracy_score(to_binary(y_true, threshold), to_binary(y_pred, threshold))


def rmse_value(y_true, y_pred):
	return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_training_history(path):
	try:
		with open(path, 'r', encoding='utf-8') as f:
			data = json.load(f)
			if isinstance(data, list):
				return data
	except Exception:
		pass
	return []

# 4. Split the data into Training and Testing sets
# We use 80% of data to teach the model, and keep 20% hidden to test it.
flare_threshold = 5
y_binary = to_binary(y, flare_threshold)
try:
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y_binary
	)
except ValueError:
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42
	)

# 5. Train the Model (The "Brain")
print("Training the Random Forest model (this might take a few seconds)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Test the Model's Accuracy
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = rmse_value(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Model successfully trained! Average Prediction Error: +/- {mae:.2f} symptom points.")

# Extra classification-style evaluation for flare-up detection
# A score >= 5 is treated as "flare-up" for confusion matrix and recall.
y_test_binary = to_binary(y_test, flare_threshold)
y_pred_binary = to_binary(predictions, flare_threshold)

cm = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1])
recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
balanced_acc = balanced_accuracy_score(y_test_binary, y_pred_binary)

if len(np.unique(y_test_binary)) > 1:
	roc_auc = roc_auc_score(y_test_binary, predictions)
else:
	roc_auc = None

class_counts = y_binary.value_counts()
min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
cv_metrics = None
if min_class_count >= 2:
	cv_splits = min(5, min_class_count)
	skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
	# Use a lighter model for CV to keep runtime manageable while preserving signal.
	cv_model = RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1)
	scoring = {
		"mae": make_scorer(mean_absolute_error, greater_is_better=False),
		"rmse": make_scorer(rmse_value, greater_is_better=False),
		"r2": make_scorer(r2_score),
		"recall": make_scorer(recall_flare, threshold=flare_threshold),
		"precision": make_scorer(precision_flare, threshold=flare_threshold),
		"f1": make_scorer(f1_flare, threshold=flare_threshold),
		"balanced_accuracy": make_scorer(balanced_accuracy_flare, threshold=flare_threshold),
	}
	print(f"Running {cv_splits}-fold cross-validation...")
	try:
		cv_results = cross_validate(
			cv_model,
			X,
			y,
			cv=skf,
			scoring=scoring,
			n_jobs=1,
			return_train_score=False,
		)
		cv_metrics = {
			"folds": cv_splits,
			"mae_mean": round(float(-cv_results["test_mae"].mean()), 4),
			"rmse_mean": round(float(-cv_results["test_rmse"].mean()), 4),
			"r2_mean": round(float(cv_results["test_r2"].mean()), 4),
			"recall_mean": round(float(cv_results["test_recall"].mean()), 4),
			"precision_mean": round(float(cv_results["test_precision"].mean()), 4),
			"f1_mean": round(float(cv_results["test_f1"].mean()), 4),
			"balanced_accuracy_mean": round(float(cv_results["test_balanced_accuracy"].mean()), 4),
		}
	except KeyboardInterrupt:
		print("Cross-validation interrupted by user. Continuing without CV metrics.")
	except Exception as exc:
		print(f"Cross-validation skipped due to error: {exc}")

print("\nConfusion Matrix (rows=true, cols=pred):")
print("             Pred: No Flare  Pred: Flare")
print(f"True: No Flare      {cm[0][0]:>5}         {cm[0][1]:>5}")
print(f"True: Flare         {cm[1][0]:>5}         {cm[1][1]:>5}")
print(f"Recall (Flare class): {recall:.3f}")
print(f"Precision (Flare class): {precision:.3f}")
print(f"F1 (Flare class): {f1:.3f}")
if roc_auc is not None:
	print(f"ROC-AUC (Flare class): {roc_auc:.3f}")
if cv_metrics:
	print(f"CV MAE (mean): {cv_metrics['mae_mean']:.3f}")
	print(f"CV F1 (mean): {cv_metrics['f1_mean']:.3f}")

# Save test metrics so Flask can display model quality in the UI.
model_test_results = {
	"trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
	"sample_count": int(len(df)),
	"feature_count": int(X.shape[1]),
	"mae": round(float(mae), 3),
	"rmse": round(float(rmse), 3),
	"r2": round(float(r2), 3),
	"flare_threshold": flare_threshold,
	"recall": round(float(recall), 3),
	"precision": round(float(precision), 3),
	"f1": round(float(f1), 3),
	"balanced_accuracy": round(float(balanced_acc), 3),
	"roc_auc": round(float(roc_auc), 3) if roc_auc is not None else None,
	"confusion_matrix": {
		"tn": int(cm[0][0]),
		"fp": int(cm[0][1]),
		"fn": int(cm[1][0]),
		"tp": int(cm[1][1])
	},
	"cross_validation": cv_metrics
}

with open('model_test_results.json', 'w', encoding='utf-8') as f:
	json.dump(model_test_results, f, indent=2)
print("Model test results saved as 'model_test_results.json'.")

feature_importances = []
if hasattr(model, "feature_importances_"):
	for feature_name, importance in zip(X.columns, model.feature_importances_):
		feature_importances.append({
			"feature": feature_name,
			"importance": round(float(importance), 6)
		})
	feature_importances.sort(key=lambda item: item["importance"], reverse=True)
	feature_importances = feature_importances[:20]

feature_importance_payload = {
	"trained_at": model_test_results["trained_at"],
	"top_features": feature_importances
}
with open('model_feature_importance.json', 'w', encoding='utf-8') as f:
	json.dump(feature_importance_payload, f, indent=2)
print("Feature importance saved as 'model_feature_importance.json'.")

history_file = 'model_training_history.json'
history = load_training_history(history_file)
history.append(model_test_results)
# Keep file compact by storing only the most recent 50 runs.
history = history[-50:]
with open(history_file, 'w', encoding='utf-8') as f:
	json.dump(history, f, indent=2)
print("Training history updated in 'model_training_history.json'.")

# 7. Save the trained model to a file
joblib.dump(model, 'gut_health_model.pkl')
print("Model saved as 'gut_health_model.pkl'. Ready for Flask!")