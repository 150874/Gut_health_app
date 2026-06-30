import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
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

if 'Food_Name' in df.columns:
	df['Food_Name'] = (
		df['Food_Name']
		.fillna('Unknown Food')
		.astype(str)
		.str.strip()
		.str.lower()
	)
else:
	df['Food_Name'] = 'unknown food'

# 2. Select Features (X) and Target (y)
# We drop User_ID and Timestamp because the model shouldn't learn from names/dates.
# We also drop the Target variable from X.
X_raw = df.drop(columns=['User_ID', 'Timestamp', 'Symptom_Flare_Up_Score'], errors='ignore')
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


def compute_threshold_metrics(y_true, predictions, thresholds):
	rows = []
	for threshold in thresholds:
		y_true_binary = to_binary(y_true, threshold)
		y_pred_binary = to_binary(predictions, threshold)
		rows.append({
			"threshold": round(float(threshold), 2),
			"recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
			"precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
			"f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
			"balanced_accuracy": float(balanced_accuracy_score(y_true_binary, y_pred_binary)),
		})
	return rows


def choose_threshold(metrics_rows, precision_floor=0.65):
	if not metrics_rows:
		return 5.0, []

	candidates = [row for row in metrics_rows if row["precision"] >= precision_floor]
	if candidates:
		candidates.sort(key=lambda item: (item["recall"], item["f1"], item["balanced_accuracy"]), reverse=True)
		picked = candidates[0]
	else:
		fallback = sorted(metrics_rows, key=lambda item: (item["f1"], item["balanced_accuracy"], item["recall"]), reverse=True)
		picked = fallback[0]

	leaderboard = sorted(metrics_rows, key=lambda item: (item["f1"], item["recall"], item["precision"]), reverse=True)[:10]
	return float(picked["threshold"]), leaderboard


def build_calibration_bins(y_true_binary, predictions, bin_width=1.0, min_samples=30):
	bins = []
	start = 0.0
	while start < 10.0:
		end = round(start + bin_width, 2)
		if end >= 10.0:
			mask = (predictions >= start) & (predictions <= 10.0)
		else:
			mask = (predictions >= start) & (predictions < end)

		indices = np.where(mask)[0]
		samples = int(len(indices))
		if samples >= min_samples:
			true_slice = y_true_binary.iloc[indices] if hasattr(y_true_binary, 'iloc') else y_true_binary[indices]
			observed = float(np.mean(true_slice)) if samples > 0 else 0.0
			bins.append({
				"min_score": round(start, 2),
				"max_score": round(min(end, 10.0), 2),
				"observed_flare_rate": round(observed, 4),
				"sample_count": samples,
			})
		start = end
	return bins


def temporal_validation_metrics(model, X_matrix, y_values, raw_df, threshold):
	if 'Timestamp' not in raw_df.columns:
		return None

	timestamps = pd.to_datetime(raw_df['Timestamp'], errors='coerce')
	valid_mask = timestamps.notna()
	if int(valid_mask.sum()) < 500:
		return None

	ordered_idx = timestamps[valid_mask].sort_values().index
	if len(ordered_idx) < 500:
		return None

	split_at = int(len(ordered_idx) * 0.8)
	if split_at <= 0 or split_at >= len(ordered_idx):
		return None

	train_idx = ordered_idx[:split_at]
	test_idx = ordered_idx[split_at:]

	X_train_t = X_matrix.loc[train_idx]
	X_test_t = X_matrix.loc[test_idx]
	y_train_t = y_values.loc[train_idx]
	y_test_t = y_values.loc[test_idx]

	temp_model = clone(model)
	if isinstance(temp_model, (RandomForestRegressor, ExtraTreesRegressor)):
		temp_model.set_params(n_estimators=80)

	temp_model.fit(X_train_t, y_train_t)
	preds = temp_model.predict(X_test_t)

	y_true_binary = to_binary(y_test_t, threshold)
	y_pred_binary = to_binary(preds, threshold)

	metrics = {
		"test_samples": int(len(y_test_t)),
		"mae": round(float(mean_absolute_error(y_test_t, preds)), 4),
		"rmse": round(float(rmse_value(y_test_t, preds)), 4),
		"r2": round(float(r2_score(y_test_t, preds)), 4),
		"recall": round(float(recall_score(y_true_binary, y_pred_binary, zero_division=0)), 4),
		"precision": round(float(precision_score(y_true_binary, y_pred_binary, zero_division=0)), 4),
		"f1": round(float(f1_score(y_true_binary, y_pred_binary, zero_division=0)), 4),
		"balanced_accuracy": round(float(balanced_accuracy_score(y_true_binary, y_pred_binary)), 4),
	}

	if len(np.unique(y_true_binary)) > 1:
		metrics["roc_auc"] = round(float(roc_auc_score(y_true_binary, preds)), 4)
	else:
		metrics["roc_auc"] = None

	return metrics


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test, flare_threshold):
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)

	mae = mean_absolute_error(y_test, predictions)
	rmse = rmse_value(y_test, predictions)
	r2 = r2_score(y_test, predictions)

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

	return {
		"name": model_name,
		"model": model,
		"predictions": predictions,
		"mae": float(mae),
		"rmse": float(rmse),
		"r2": float(r2),
		"recall": float(recall),
		"precision": float(precision),
		"f1": float(f1),
		"balanced_accuracy": float(balanced_acc),
		"roc_auc": float(roc_auc) if roc_auc is not None else None,
		"confusion_matrix": {
			"tn": int(cm[0][0]),
			"fp": int(cm[0][1]),
			"fn": int(cm[1][0]),
			"tp": int(cm[1][1]),
		},
	}


def selection_key(result):
	# Prioritize flare-up detection quality first, then overall error.
	return (
		result["f1"],
		result["balanced_accuracy"],
		result["recall"],
		-result["mae"],
	)


def extract_feature_importances(model, feature_names):
	if hasattr(model, "feature_importances_"):
		values = np.asarray(model.feature_importances_, dtype=float)
	elif hasattr(model, "coef_"):
		coef = np.asarray(model.coef_, dtype=float)
		values = np.abs(coef)
	else:
		return []

	if values.ndim > 1:
		values = values.ravel()

	total = float(values.sum())
	if total > 0:
		values = values / total

	items = []
	for feature_name, importance in zip(feature_names, values):
		items.append({
			"feature": feature_name,
			"importance": round(float(importance), 6)
		})

	items.sort(key=lambda item: item["importance"], reverse=True)
	return items[:20]

# 4. Split the data into Training and Testing sets
# We use 80% of data to teach the model, and keep 20% hidden to test it.
base_flare_threshold = 5.0
y_binary = to_binary(y, base_flare_threshold)
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

# 5. Train Multiple Models and Select the Best
candidate_models = {
	"RandomForest_160": RandomForestRegressor(n_estimators=160, random_state=42, n_jobs=-1),
	"RandomForest_260": RandomForestRegressor(n_estimators=260, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1),
	"ExtraTrees_220": ExtraTreesRegressor(n_estimators=220, random_state=42, n_jobs=-1),
	"ExtraTrees_320": ExtraTreesRegressor(n_estimators=320, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1),
	"GradientBoosting_120": GradientBoostingRegressor(n_estimators=120, random_state=42),
	"GradientBoosting_180": GradientBoostingRegressor(n_estimators=180, learning_rate=0.06, random_state=42),
	"LinearRegression": LinearRegression(),
}

results = []
print("Training and evaluating model candidates...")
for model_name, estimator in candidate_models.items():
	print(f" - Training {model_name}...")
	try:
		result = evaluate_model(model_name, estimator, X_train, y_train, X_test, y_test, base_flare_threshold)
		results.append(result)
		print(
			f"   Done | MAE={result['mae']:.3f} | RMSE={result['rmse']:.3f} | "
			f"F1={result['f1']:.3f} | Recall={result['recall']:.3f}"
		)
	except Exception as exc:
		print(f"   Skipped {model_name} due to error: {exc}")

if not results:
	raise RuntimeError("No model candidate trained successfully.")

results.sort(key=selection_key, reverse=True)
best = results[0]
model = best["model"]
predictions = best["predictions"]

threshold_grid = np.arange(3.5, 6.6, 0.1)
threshold_metrics = compute_threshold_metrics(y_test, predictions, threshold_grid)
optimized_threshold, threshold_leaderboard = choose_threshold(threshold_metrics, precision_floor=0.65)

y_test_binary = to_binary(y_test, optimized_threshold)
y_pred_binary = to_binary(predictions, optimized_threshold)
cm_array = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1])

mae = mean_absolute_error(y_test, predictions)
rmse = rmse_value(y_test, predictions)
r2 = r2_score(y_test, predictions)
recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
balanced_acc = balanced_accuracy_score(y_test_binary, y_pred_binary)
roc_auc = best["roc_auc"]
cm = {
	"tn": int(cm_array[0][0]),
	"fp": int(cm_array[0][1]),
	"fn": int(cm_array[1][0]),
	"tp": int(cm_array[1][1]),
}

calibration_bins = build_calibration_bins(y_test_binary, predictions, bin_width=1.0, min_samples=35)
temporal_validation = temporal_validation_metrics(model, X, y, df, optimized_threshold)

print(
	f"Best model selected: {best['name']} "
	f"(F1={f1:.3f}, Recall={recall:.3f}, MAE={mae:.3f}, threshold={optimized_threshold:.2f})"
)

class_counts = y_binary.value_counts()
min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
cv_metrics = None
if min_class_count >= 2:
	cv_splits = min(5, min_class_count)
	skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
	# Use selected model for CV, with lighter settings for tree ensembles to keep runtime manageable.
	cv_model = clone(model)
	if isinstance(cv_model, (RandomForestRegressor, ExtraTreesRegressor)):
		cv_model.set_params(n_estimators=80)
	scoring = {
		"mae": make_scorer(mean_absolute_error, greater_is_better=False),
		"rmse": make_scorer(rmse_value, greater_is_better=False),
		"r2": make_scorer(r2_score),
		"recall": make_scorer(recall_flare, threshold=optimized_threshold),
		"precision": make_scorer(precision_flare, threshold=optimized_threshold),
		"f1": make_scorer(f1_flare, threshold=optimized_threshold),
		"balanced_accuracy": make_scorer(balanced_accuracy_flare, threshold=optimized_threshold),
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
			"model": best["name"],
			"folds": cv_splits,
			"mae_mean": round(float(-cv_results["test_mae"].mean()), 4),
			"rmse_mean": round(float(-cv_results["test_rmse"].mean()), 4),
			"r2_mean": round(float(cv_results["test_r2"].mean()), 4),
			"recall_std": round(float(cv_results["test_recall"].std()), 4),
			"f1_std": round(float(cv_results["test_f1"].std()), 4),
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
print(f"True: No Flare      {cm['tn']:>5}         {cm['fp']:>5}")
print(f"True: Flare         {cm['fn']:>5}         {cm['tp']:>5}")
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
	"model_name": best["name"],
	"selection_criteria": "max_f1_then_balanced_accuracy_then_recall_then_min_mae",
	"sample_count": int(len(df)),
	"feature_count": int(X.shape[1]),
	"mae": round(float(mae), 3),
	"rmse": round(float(rmse), 3),
	"r2": round(float(r2), 3),
	"flare_threshold": base_flare_threshold,
	"optimized_flare_threshold": round(float(optimized_threshold), 2),
	"recall": round(float(recall), 3),
	"precision": round(float(precision), 3),
	"f1": round(float(f1), 3),
	"balanced_accuracy": round(float(balanced_acc), 3),
	"roc_auc": round(float(roc_auc), 3) if roc_auc is not None else None,
	"confusion_matrix": cm,
	"threshold_tuning": {
		"strategy": "maximize_recall_with_precision_floor",
		"target_precision_floor": 0.65,
		"selected_threshold": round(float(optimized_threshold), 2),
		"leaderboard": [
			{
				"threshold": row["threshold"],
				"recall": round(float(row["recall"]), 3),
				"precision": round(float(row["precision"]), 3),
				"f1": round(float(row["f1"]), 3),
				"balanced_accuracy": round(float(row["balanced_accuracy"]), 3),
			}
			for row in threshold_leaderboard
		],
	},
	"calibration_bins": calibration_bins,
	"temporal_validation": temporal_validation,
	"model_candidates": [
		{
			"model_name": item["name"],
			"mae": round(float(item["mae"]), 3),
			"rmse": round(float(item["rmse"]), 3),
			"r2": round(float(item["r2"]), 3),
			"recall": round(float(item["recall"]), 3),
			"precision": round(float(item["precision"]), 3),
			"f1": round(float(item["f1"]), 3),
			"balanced_accuracy": round(float(item["balanced_accuracy"]), 3),
			"roc_auc": round(float(item["roc_auc"]), 3) if item["roc_auc"] is not None else None,
		}
		for item in results
	],
	"cross_validation": cv_metrics
}

with open('model_test_results.json', 'w', encoding='utf-8') as f:
	json.dump(model_test_results, f, indent=2)
print("Model test results saved as 'model_test_results.json'.")

feature_importances = extract_feature_importances(model, X.columns)

feature_importance_payload = {
	"trained_at": model_test_results["trained_at"],
	"model_name": best["name"],
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