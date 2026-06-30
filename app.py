
from flask import Flask, render_template, request, redirect, url_for, flash, session, g
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import date, datetime, timedelta
from contextlib import contextmanager
import logging
import time
import joblib
import pandas as pd
import json
import os
import secrets
from admin_routes import register_admin_routes


def load_env_file(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Keep shell-exported values as highest priority.
            if key and key not in os.environ:
                os.environ[key] = value

# ---------------------------------------------------
# APP SETUP
# ---------------------------------------------------
if os.path.exists(".env"):
    load_env_file(".env")
elif os.path.exists(".env.example"):
    load_env_file(".env.example")

app = Flask(__name__)

# --- MACHINE LEARNING SETUP ---
try:
    print("Loading Gut Health ML Model...")
    flare_up_model = joblib.load('gut_health_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load ML model. {e}")
    flare_up_model = None

try:
    with open('model_test_results.json', 'r', encoding='utf-8') as f:
        model_test_results = json.load(f)
except Exception:
    model_test_results = None


def load_training_history():
    try:
        with open('model_training_history.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def load_feature_importance():
    try:
        with open('model_feature_importance.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {"trained_at": None, "top_features": []}


def get_model_test_results():
    global model_test_results
    try:
        with open('model_test_results.json', 'r', encoding='utf-8') as f:
            model_test_results = json.load(f)
    except Exception:
        pass
    return model_test_results


def get_prediction_threshold(default=5.0):
    metrics = get_model_test_results() or {}
    threshold = metrics.get("optimized_flare_threshold", metrics.get("flare_threshold", default))
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return float(default)


def get_runtime_health():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT route, duration_ms, status_code, created_at "
            "FROM request_metrics "
            "WHERE created_at >= datetime('now', '-24 hours') "
            "ORDER BY id DESC"
        ).fetchall()

        slow_rows = conn.execute(
            "SELECT route, COUNT(*) AS hits, ROUND(AVG(duration_ms), 2) AS avg_ms "
            "FROM request_metrics "
            "WHERE created_at >= datetime('now', '-24 hours') "
            "GROUP BY route "
            "ORDER BY avg_ms DESC, hits DESC "
            "LIMIT 5"
        ).fetchall()

    durations = sorted([float(row["duration_ms"] or 0.0) for row in rows])

    def percentile(values, p):
        if not values:
            return 0.0
        idx = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
        return float(values[idx])

    success_count = sum(1 for row in rows if 200 <= int(row["status_code"] or 0) < 400)
    error_count = sum(1 for row in rows if int(row["status_code"] or 0) >= 400)
    total = len(rows)

    return {
        "window_hours": 24,
        "request_count": total,
        "success_rate": round((success_count / total), 4) if total else None,
        "error_rate": round((error_count / total), 4) if total else None,
        "latency_ms": {
            "p50": round(percentile(durations, 50), 2),
            "p95": round(percentile(durations, 95), 2),
            "avg": round((sum(durations) / len(durations)), 2) if durations else 0.0,
        },
        "slow_routes": [
            {
                "route": row["route"],
                "hits": int(row["hits"] or 0),
                "avg_ms": float(row["avg_ms"] or 0.0),
            }
            for row in slow_rows
        ],
    }

app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
if not os.getenv("FLASK_SECRET_KEY"):
    logging.warning("FLASK_SECRET_KEY is not set. Using an ephemeral key for this run.")

logging.basicConfig(level=logging.INFO)

# Move route definitions below app initialization
@app.route("/delete_med/<int:med_id>", methods=["POST"])
def delete_med(med_id):
    if not is_logged_in():
        return redirect(url_for("login"))
    with get_db() as conn:
        conn.execute("DELETE FROM medications WHERE id = ? AND user_id = ?", (med_id, session["user_id"]))
    flash("Medication deleted.")
    return redirect(url_for("view_meds"))

@app.route("/delete_symptom/<int:symptom_id>", methods=["POST"])
def delete_symptom(symptom_id):
    if not is_logged_in():
        return redirect(url_for("login"))
    with get_db() as conn:
        conn.execute("DELETE FROM symptoms WHERE id = ? AND user_id = ?", (symptom_id, session["user_id"]))
    flash("Symptom deleted.")
    return redirect(url_for("view_symptoms"))

@app.route("/delete_food_log/<int:log_id>", methods=["POST"])
def delete_food_log(log_id):
    if not is_logged_in():
        return redirect(url_for("login"))
    with get_db() as conn:
        conn.execute("DELETE FROM food_logs WHERE id = ? AND user_id = ?", (log_id, session["user_id"]))
    flash("Food log deleted.")
    return redirect(url_for("diet_checker"))

DB_PATH = "database.db"

# Map user condition to the correct database column for safety checks
CONDITION_COLUMN_MAP = {
    "gastritis": "gastritis_status",
    "gerd": "gerd_status",
    "hpylori": "hpylori_status"
}

MODEL_CONDITION_MAP = {
    "gastritis": "Gastritis",
    "gerd": "GERD",
    "hpylori": "H. Pylori",
    "general": "General"
}


def get_admin_emails():
    raw = os.getenv("ADMIN_EMAILS", "")
    return {email.strip().lower() for email in raw.split(",") if email.strip()}


def is_admin_email(email):
    return email.strip().lower() in get_admin_emails()


def to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_food_name(value):
    if value is None:
        return "unknown food"
    text = str(value).strip().lower()
    return text if text else "unknown food"


def estimate_pral_score(food_name, meal_type="Lunch"):
    normalized_food = normalize_food_name(food_name)
    known_score = lookup_pral_score(normalized_food)
    if known_score is not None:
        return float(known_score), "knowledge_base"

    keyword_scores = {
        "fried": 6.0,
        "pizza": 5.0,
        "burger": 6.3,
        "sausage": 6.2,
        "bacon": 6.0,
        "beef": 5.2,
        "chicken": 2.8,
        "cheese": 4.5,
        "tomato": 3.8,
        "citrus": 3.4,
        "pepper": 3.0,
        "spicy": 4.2,
        "rice": 1.0,
        "oat": -0.8,
        "banana": -5.5,
        "salad": -2.8,
        "lentil": -3.0,
        "bean": -2.4,
        "vegetable": -2.6,
        "tofu": -2.2,
        "fruit": -1.8,
    }

    matched_scores = [value for keyword, value in keyword_scores.items() if keyword in normalized_food]
    if matched_scores:
        return round(float(sum(matched_scores) / len(matched_scores)), 2), "estimated_from_food"

    meal_type_defaults = {
        "breakfast": 0.7,
        "lunch": 1.4,
        "dinner": 2.0,
    }
    fallback = meal_type_defaults.get(str(meal_type).strip().lower(), 1.2)
    return float(fallback), "estimated_default"


def suggest_alternative_meals(condition_value, meal_type="Lunch", exclude_food="", limit=3):
    column = CONDITION_COLUMN_MAP.get(condition_value, "gastritis_status")
    normalized_exclude = normalize_food_name(exclude_food)

    with get_db() as conn:
        rows = conn.execute(
            f"SELECT fr.food_name, COALESCE(fp.pral_score, 0) AS pral_score "
            f"FROM food_rules fr "
            f"LEFT JOIN food_pral fp ON LOWER(fp.food_name) = LOWER(fr.food_name) "
            f"WHERE LOWER(fr.{column}) = 'safe' AND LOWER(fr.food_name) != ? "
            f"ORDER BY ABS(COALESCE(fp.pral_score, 0)) ASC, LOWER(fr.food_name) ASC LIMIT ?",
            (normalized_exclude, limit)
        ).fetchall()

    alternatives = []
    for row in rows:
        reason = "Generally gentle for this condition"
        if row["pral_score"] <= -1:
            reason = "Lower acid load option"
        elif row["pral_score"] <= 1.5:
            reason = "Mild acid load option"
        alternatives.append({
            "food": row["food_name"],
            "reason": reason,
            "pral_score": round(float(row["pral_score"]), 2),
            "meal_type": meal_type,
        })
    return alternatives


def build_followup_answer(question, context):
    q = (question or "").strip().lower()
    if not q:
        return "Ask me anything about this meal, like why the risk is high, better options, or how to lower risk next time."

    why_points = context.get("why_points") or []
    alternatives = context.get("alternatives") or []
    risk_level = context.get("risk_level", "moderate")
    food_name = context.get("food_name", "this meal")

    if "why" in q or "reason" in q:
        if why_points:
            return "Main reasons: " + "; ".join(why_points[:3])
        return "The risk is based on your meal acid load, hydration, stress level, and condition profile."

    if "alternative" in q or "instead" in q or "replace" in q:
        if alternatives:
            picks = ", ".join(item["food"] for item in alternatives[:3])
            return f"Better options for your condition: {picks}."
        return "Try a lower-acid, less spicy, and better-hydrated meal option for your condition."

    if "lower" in q and "risk" in q:
        return "To lower risk: improve hydration (>= 300 ml), reduce acidic or spicy items, and keep stress low during meals."

    if "safe" in q:
        if risk_level == "low":
            return f"{food_name.capitalize()} appears relatively safe for now, but continue monitoring symptoms."
        return f"{food_name.capitalize()} is not the safest pick right now based on your current inputs."

    return "Based on this meal, focus on lower acid load, better hydration, and calmer eating conditions. Ask for alternatives for specific swaps."


def build_prediction_why_points(user_data, condition_value, food_name, looked_up_pral=None):
    points = []
    meal_pral = to_float(user_data.get("Meal_PRAL_Score"), 0)
    water_ml = int(to_float(user_data.get("Water_Consumed_ml"), 0))
    stress_level = str(user_data.get("Stress_At_Meal", "Moderate")).strip().lower()

    if meal_pral >= 5:
        points.append("High acid load from this meal (high PRAL score).")
    elif meal_pral >= 2:
        points.append("Moderate acid load that may trigger symptoms in sensitive users.")
    elif meal_pral <= -1.5:
        points.append("Lower acid load from this meal profile.")

    acidic_keywords = ["tomato", "citrus", "orange", "lemon", "spicy", "pepper", "fried", "pizza"]
    if any(word in food_name for word in acidic_keywords):
        points.append("This food may be acidic or irritating for your condition.")

    if looked_up_pral is not None:
        points.append("PRAL value was matched from the food knowledge base.")

    if water_ml < 200:
        points.append("Low water intake at this meal may increase irritation risk.")
    elif water_ml >= 450:
        points.append("Good hydration may help reduce discomfort risk.")

    if stress_level == "high":
        points.append("High stress level can increase gut sensitivity and flare-up risk.")
    elif stress_level == "moderate":
        points.append("Moderate stress can still influence digestive symptoms.")

    if condition_value == "gastritis":
        points.append("For gastritis, irritating meals may inflame the stomach lining.")
    elif condition_value == "gerd":
        points.append("For GERD, trigger meals can increase reflux and heartburn symptoms.")
    elif condition_value == "hpylori":
        points.append("For H. pylori, irritating meals may worsen upper abdominal discomfort.")

    if not points:
        points.append("No strong risk triggers were detected from the current meal inputs.")

    return points[:5]


def get_recent_symptom_burden(user_id):
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN timestamp >= datetime('now', '-7 day') THEN 1 ELSE 0 END) AS recent_count,
                SUM(CASE WHEN timestamp >= datetime('now', '-14 day') AND timestamp < datetime('now', '-7 day') THEN 1 ELSE 0 END) AS previous_count,
                SUM(CASE WHEN timestamp >= datetime('now', '-7 day') AND LOWER(severity) = 'severe' THEN 1 ELSE 0 END) AS severe_count,
                AVG(CASE WHEN timestamp >= datetime('now', '-7 day') THEN
                    CASE LOWER(severity)
                        WHEN 'mild' THEN 1.0
                        WHEN 'moderate' THEN 2.0
                        WHEN 'severe' THEN 3.0
                        ELSE 1.0
                    END
                END) AS avg_severity
            FROM symptoms
            WHERE user_id = ?
            """,
            (user_id,)
        ).fetchone()

    recent_count = int(row["recent_count"] or 0)
    previous_count = int(row["previous_count"] or 0)
    severe_count = int(row["severe_count"] or 0)
    avg_severity = float(row["avg_severity"] or 0.0)

    trend_delta = recent_count - previous_count
    burden_score = (recent_count * 0.28) + (severe_count * 0.9) + (avg_severity * 1.1) + max(0.0, trend_delta * 0.25)
    burden_score = max(0.0, min(10.0, burden_score))

    if burden_score >= 7.5:
        level = "high"
    elif burden_score >= 4.0:
        level = "moderate"
    else:
        level = "low"

    return {
        "recent_count": recent_count,
        "previous_count": previous_count,
        "severe_count": severe_count,
        "avg_severity": round(avg_severity, 2),
        "trend_delta": trend_delta,
        "burden_score": round(float(burden_score), 2),
        "burden_level": level,
    }


def calibrate_confidence_from_bins(score):
    metrics = get_model_test_results() or {}
    bins = metrics.get("calibration_bins")
    if not isinstance(bins, list):
        return None
    for row in bins:
        try:
            min_score = float(row.get("min_score", -1))
            max_score = float(row.get("max_score", -1))
            if min_score <= score <= max_score:
                observed = float(row.get("observed_flare_rate", 0.5))
                certainty = min(1.0, abs(observed - 0.5) * 2.0)
                return max(0.0, min(1.0, 0.52 + (0.43 * certainty)))
        except (TypeError, ValueError):
            continue
    return None


def build_prediction_result(score):
    raw_score = float(score)
    score = round(raw_score, 1)
    bounded_score = max(0.0, min(10.0, raw_score))

    threshold = get_prediction_threshold(default=5.0)
    moderate_threshold = max(3.0, min(8.0, threshold))
    high_threshold = max(moderate_threshold + 1.5, min(9.2, moderate_threshold + 2.3))

    # Confidence is higher when the score is farther from decision boundaries.
    distance_to_boundary = min(abs(bounded_score - moderate_threshold), abs(bounded_score - high_threshold))
    boundary_confidence = min(distance_to_boundary / 3.0, 1.0)
    confidence_score = 0.55 + (0.40 * boundary_confidence)

    calibrated_confidence = calibrate_confidence_from_bins(bounded_score)
    if calibrated_confidence is not None:
        confidence_score = (confidence_score * 0.65) + (calibrated_confidence * 0.35)

    # Calibrate by recent model quality when available.
    quality_signal = None
    model_metrics = get_model_test_results() or {}
    quality_signal = model_metrics.get("balanced_accuracy")
    if quality_signal is None:
        quality_signal = model_metrics.get("f1")
    if isinstance(quality_signal, (int, float)):
        quality_scaled = max(0.0, min(1.0, float(quality_signal)))
        confidence_score = (confidence_score * 0.7) + (quality_scaled * 0.3)

    confidence_score = max(0.0, min(1.0, confidence_score))
    confidence_percent = int(round(confidence_score * 100))
    if confidence_percent >= 82:
        confidence_label = "High"
    elif confidence_percent >= 68:
        confidence_label = "Moderate"
    else:
        confidence_label = "Low"

    if score >= high_threshold:
        level = "high"
        title = "DANGER!"
        message = "High probability of a flare-up. Consider drinking more water or changing the meal."
        recommendation = "Avoid This Meal"
        recommendation_reason = "Your predicted flare-up risk is high for your current condition profile."
    elif score >= moderate_threshold:
        level = "moderate"
        title = "Warning."
        message = "Moderate risk. Proceed with caution."
        recommendation = "Use Caution"
        recommendation_reason = "Your risk is moderate. Portion size, hydration, and stress can change the outcome."
    else:
        level = "low"
        title = "All Clear!"
        message = "This meal profile looks safe for your gut health."
        recommendation = "Safe To Try"
        recommendation_reason = "Your predicted flare-up risk is low for this meal profile."

    if confidence_label == "Low":
        recommendation_reason += " Confidence is low, so monitor symptoms after eating."

    return {
        "score": score,
        "level": level,
        "title": title,
        "message": message,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "confidence_percent": confidence_percent,
        "confidence_label": confidence_label,
        "calibration_hint": (
            "Confidence blended with empirical score-to-flare calibration bins."
            if calibrated_confidence is not None
            else None
        ),
        "decision_threshold": round(float(moderate_threshold), 2),
        "high_risk_threshold": round(float(high_threshold), 2),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }


def assess_prediction_quality(food_name, pral_source, model_columns):
    food_feature_name = f"Food_Name_{food_name}"
    has_food_feature = food_feature_name in model_columns

    if has_food_feature and pral_source == "knowledge_base":
        return {
            "quality": "high",
            "is_uncertain": False,
            "reason": "Food exists in trained vocabulary and PRAL came from knowledge base."
        }

    if has_food_feature or pral_source == "estimated_from_food":
        return {
            "quality": "medium",
            "is_uncertain": False,
            "reason": "Prediction uses partial food signal with estimated acidity profile."
        }

    return {
        "quality": "low",
        "is_uncertain": True,
        "reason": "Food is not in trained vocabulary and PRAL is a fallback estimate."
    }

# ---------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Database Error: {e}")
        raise
    finally:
        conn.close()


def log_admin_action(action_type, target, details=""):
    admin_id = session.get("user_id")
    if not admin_id:
        return
    with get_db() as conn:
        conn.execute(
            "INSERT INTO admin_audit_logs (admin_user_id, action_type, target, details) VALUES (?, ?, ?, ?)",
            (admin_id, action_type, target, details)
        )

# ---------------------------------------------------
# INITIALIZATION & SEEDING
# ---------------------------------------------------
def init_db():
    with get_db() as conn:
        c = conn.cursor()
        # Create all tables (Users, Food Rules, Logs, Symptoms, Knowledge, Medications)
        c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT UNIQUE, password TEXT, condition TEXT, age REAL DEFAULT 30, bmi REAL DEFAULT 24.5, h_pylori_result TEXT DEFAULT 'Negative', is_admin INTEGER DEFAULT 0, is_active INTEGER DEFAULT 1)")
        c.execute("CREATE TABLE IF NOT EXISTS food_rules (id INTEGER PRIMARY KEY AUTOINCREMENT, food_name TEXT UNIQUE, gastritis_status TEXT, gerd_status TEXT, hpylori_status TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS food_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, food_name TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        c.execute("CREATE TABLE IF NOT EXISTS symptoms (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, symptom TEXT, severity TEXT, note TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        c.execute("CREATE TABLE IF NOT EXISTS symptom_knowledge (id INTEGER PRIMARY KEY AUTOINCREMENT, symptom_name TEXT UNIQUE, duration TEXT, seek_help TEXT, tips TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS medications (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, name TEXT, dosage TEXT, time TEXT, times_per_day INTEGER DEFAULT 1, duration_days INTEGER DEFAULT 14, instructions TEXT, taken_today INTEGER DEFAULT 0, last_taken_date TEXT, total_taken INTEGER DEFAULT 0)")
        c.execute("CREATE TABLE IF NOT EXISTS food_pral (id INTEGER PRIMARY KEY AUTOINCREMENT, food_name TEXT UNIQUE, pral_score REAL, source TEXT DEFAULT 'manual', updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        c.execute("CREATE TABLE IF NOT EXISTS admin_audit_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, admin_user_id INTEGER, action_type TEXT, target TEXT, details TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        c.execute("CREATE TABLE IF NOT EXISTS request_metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, route TEXT, method TEXT, duration_ms REAL, status_code INTEGER, user_id INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")

        c.execute("CREATE INDEX IF NOT EXISTS idx_food_logs_user_timestamp ON food_logs(user_id, timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_user_timestamp ON symptoms(user_id, timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_request_metrics_created_at ON request_metrics(created_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_request_metrics_route ON request_metrics(route)")

            # --- Migration: Add total_taken column if missing ---
        c.execute("PRAGMA table_info(medications)")
        columns = [row[1] for row in c.fetchall()]
        if 'total_taken' not in columns:
                c.execute("ALTER TABLE medications ADD COLUMN total_taken INTEGER DEFAULT 0")

        # --- Migration: Add is_admin column if missing ---
        c.execute("PRAGMA table_info(users)")
        user_columns = [row[1] for row in c.fetchall()]
        if 'is_admin' not in user_columns:
            c.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
        if 'is_active' not in user_columns:
            c.execute("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1")
        if 'age' not in user_columns:
            c.execute("ALTER TABLE users ADD COLUMN age REAL DEFAULT 30")
        if 'bmi' not in user_columns:
            c.execute("ALTER TABLE users ADD COLUMN bmi REAL DEFAULT 24.5")
        if 'h_pylori_result' not in user_columns:
            c.execute("ALTER TABLE users ADD COLUMN h_pylori_result TEXT DEFAULT 'Negative'")

        # Seed data if empty
        c.execute("SELECT COUNT(*) FROM food_rules")
        if c.fetchone()[0] == 0:
            foods = [("Coffee", "avoid", "avoid", "avoid"), ("Banana", "safe", "safe", "safe"), ("Lemon", "safe", "avoid", "neutral"), ("Oatmeal", "safe", "safe", "safe")]
            c.executemany("INSERT INTO food_rules (food_name, gastritis_status, gerd_status, hpylori_status) VALUES (?, ?, ?, ?)", foods)

        c.execute("SELECT COUNT(*) FROM food_pral")
        if c.fetchone()[0] == 0:
            pral_seed = [
                ("Coffee", 4.7, "seed"),
                ("Banana", -5.5, "seed"),
                ("Lemon", -2.5, "seed"),
                ("Oatmeal", -0.8, "seed")
            ]
            c.executemany(
                "INSERT INTO food_pral (food_name, pral_score, source) VALUES (?, ?, ?)",
                pral_seed
            )


def lookup_pral_score(food_name):
    if not food_name:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT pral_score FROM food_pral WHERE LOWER(food_name) = ?",
            (food_name.strip().lower(),)
        ).fetchone()
    return float(row["pral_score"]) if row else None

def is_logged_in():
    return "user_id" in session


def is_admin_user():
    return bool(session.get("is_admin", False))


def require_admin():
    if not is_logged_in():
        return redirect(url_for("login"))
    if not is_admin_user():
        flash("You are not authorized to view the admin page.")
        return redirect(url_for("home"))
    return None

# ---------------------------------------------------
# ROUTES: DASHBOARD & AUTH
# ---------------------------------------------------
@app.route("/")
def home():
    if not is_logged_in():
        return redirect(url_for("login"))
    if is_admin_user():
        return redirect(url_for("admin_dashboard"))

    uid = session["user_id"]
    is_first_visit = session.pop("is_first_visit", False)
    today_str = date.today().isoformat()
    display_date = date.today().strftime("%B %d, %Y")

    with get_db() as conn:
        # 1. Meals/Checks Logged Today
        meals = conn.execute("SELECT COUNT(*) FROM food_logs WHERE user_id = ? AND date(timestamp) = ?", (uid, today_str)).fetchone()[0]
        # 2. Symptoms Logged Today
        symptoms_count = conn.execute("SELECT COUNT(*) FROM symptoms WHERE user_id = ? AND date(timestamp) = ?", (uid, today_str)).fetchone()[0]
        # 3. Meds Progress (Full Course)
        meds_data = conn.execute("SELECT times_per_day, duration_days, total_taken FROM medications WHERE user_id = ?", (uid,)).fetchall()
        total_doses = sum(m["times_per_day"] * m["duration_days"] for m in meds_data)
        taken_doses = sum(m["total_taken"] for m in meds_data)
        progress = int((taken_doses / total_doses) * 100) if total_doses > 0 else 0
        # 4. All Meds for dashboard
        meds = conn.execute("SELECT * FROM medications WHERE user_id = ? ORDER BY id DESC", (uid,)).fetchall()

    return render_template("index.html", 
                           user_name=session["user_name"], 
                           is_first_visit=is_first_visit,
                           current_date=display_date, 
                           meals=meals, 
                           meds_count=taken_doses, 
                           symptoms_count=symptoms_count, 
                           progress=progress,
                           meds=meds)


register_admin_routes(
    app,
    require_admin=require_admin,
    load_training_history=load_training_history,
    load_feature_importance=load_feature_importance,
    get_db=get_db,
    log_admin_action=log_admin_action,
    to_float=to_float,
    is_logged_in=is_logged_in,
    lookup_pral_score=lookup_pral_score,
    estimate_pral_score=estimate_pral_score,
    get_model_test_results=get_model_test_results,
    get_runtime_health=get_runtime_health,
)


@app.before_request
def start_request_timer():
    g._request_start_time = time.perf_counter()


@app.after_request
def track_request_metrics(response):
    start_time = getattr(g, "_request_start_time", None)
    if start_time is None:
        return response

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    route = request.path or "unknown"
    if route.startswith("/static"):
        return response

    user_id = session.get("user_id")
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO request_metrics (route, method, duration_ms, status_code, user_id) VALUES (?, ?, ?, ?, ?)",
            (route, request.method, float(duration_ms), int(response.status_code), user_id)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
    return response

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        hashed_pw = generate_password_hash(request.form["password"])
        email = request.form["email"].strip().lower()
        is_admin = 1 if is_admin_email(email) else 0
        age = to_float(request.form.get("age"), 30)
        bmi = to_float(request.form.get("bmi"), 24.5)
        hpylori = request.form.get("h_pylori_result", "Negative").strip() or "Negative"
        if request.form["condition"] != "hpylori":
            hpylori = "Not Applicable"
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (name, email, password, condition, age, bmi, h_pylori_result, is_admin) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (request.form["name"], email, hashed_pw, request.form["condition"], age, bmi, hpylori, is_admin)
                )
            # Mark this browser session so the next successful login can show a first-time welcome.
            session["just_signed_up"] = True
            flash("Account created!")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists.")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    first_time = bool(session.get("just_signed_up", False))
    if request.method == "POST":
        login_email = request.form["email"].strip().lower()
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (login_email,)).fetchone()
            if user and is_admin_email(login_email) and not user["is_admin"]:
                conn.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user["id"],))
                user = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        if user and not user["is_active"]:
            flash("This account is deactivated. Contact an admin.")
            return render_template("login.html", first_time=first_time)
        if user and check_password_hash(user["password"], request.form["password"]):
            just_signed_up = bool(session.pop("just_signed_up", False))
            session.update({
                "user_id": user["id"],
                "user_name": user["name"],
                "condition": user["condition"],
                "age": to_float(user["age"], 30),
                "bmi": to_float(user["bmi"], 24.5),
                "h_pylori_result": (user["h_pylori_result"] or "Negative"),
                "is_admin": bool(user["is_admin"])
            })
            session["is_first_visit"] = just_signed_up
            target_endpoint = "admin_dashboard" if session.get("is_admin") else "home"
            return redirect(url_for(target_endpoint))
        flash("Invalid credentials.")
    return render_template("login.html", first_time=first_time)

@app.route("/logout") # Matches url_for in the new templates
def logout_page():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("login"))


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if not is_logged_in():
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        condition = request.form.get("condition", "general").strip().lower()
        if condition not in CONDITION_COLUMN_MAP and condition != "general":
            condition = "general"

        age = to_float(request.form.get("age"), 30)
        bmi = to_float(request.form.get("bmi"), 24.5)
        hpylori = request.form.get("h_pylori_result", "Negative").strip() or "Negative"
        if condition != "hpylori":
            hpylori = "Not Applicable"

        with get_db() as conn:
            conn.execute(
                "UPDATE users SET name = ?, condition = ?, age = ?, bmi = ?, h_pylori_result = ? WHERE id = ?",
                (name, condition, age, bmi, hpylori, user_id)
            )
            updated = conn.execute(
                "SELECT name, condition, age, bmi, h_pylori_result, is_admin FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()

        if updated:
            session["user_name"] = updated["name"]
            session["condition"] = updated["condition"]
            session["age"] = to_float(updated["age"], 30)
            session["bmi"] = to_float(updated["bmi"], 24.5)
            session["h_pylori_result"] = (updated["h_pylori_result"] or "Negative")
            session["is_admin"] = bool(updated["is_admin"])

        flash("Profile updated successfully.")
        return redirect(url_for("profile"))

    with get_db() as conn:
        user = conn.execute(
            "SELECT name, email, condition, age, bmi, h_pylori_result FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

    if not user:
        flash("Profile not found.")
        return redirect(url_for("logout_page"))

    return render_template("profile.html", user=user)

# ---------------------------------------------------
# ROUTES: DIET CHECKER (Fixed AssertionError)
# ---------------------------------------------------
@app.route("/diet_checker")
def diet_checker():
    if not is_logged_in():
        return redirect(url_for("login"))
    
    # Get parameters passed from the /check redirect
    food = request.args.get("food")
    status = request.args.get("status")

    # If there is no active food context, clear stale prediction cards.
    if not food:
        session.pop("last_prediction", None)
        session.pop("last_prediction_context", None)
    
    with get_db() as conn:
        recent_logs = conn.execute("SELECT id, food_name, timestamp FROM food_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5", (session["user_id"],)).fetchall()
        profile_row = conn.execute(
            "SELECT age, bmi, h_pylori_result FROM users WHERE id = ?",
            (session["user_id"],)
        ).fetchone()
    user_profile = {
        "age": to_float(profile_row["age"], 30) if profile_row else 30,
        "bmi": to_float(profile_row["bmi"], 24.5) if profile_row else 24.5,
        "h_pylori_result": (profile_row["h_pylori_result"] if profile_row else "Negative") or "Negative"
    }
    return render_template("diet_checker.html", 
                           food=food, 
                           status=status, 
                           recent_logs=recent_logs, 
                           prediction_result=session.get("last_prediction"),
                           condition=session["condition"],
                           user_profile=user_profile)

@app.route("/check", methods=["POST"])
def check():
    if not is_logged_in():
        return redirect(url_for("login"))

    food_input = request.form["food"].strip()
    if not food_input:
        flash("Please enter a food name.")
        return redirect(url_for("diet_checker"))

    food_query = food_input.lower()
    condition = session["condition"]
    column = CONDITION_COLUMN_MAP.get(condition, "gastritis_status")

    with get_db() as conn:
        row = conn.execute(f"SELECT food_name, {column} AS status FROM food_rules WHERE LOWER(food_name) = ?", (food_query,)).fetchone()

        if row:
            res_food, res_status = row["food_name"], row["status"]
        else:
            res_food, res_status = food_input, "unknown"

        # Always save what the user checked so history is complete, even for unknown foods.
        conn.execute("INSERT INTO food_logs (user_id, food_name) VALUES (?, ?)", (session["user_id"], res_food))

    # Redirect to the diet_checker page with result data in the URL
    return redirect(url_for("diet_checker", food=res_food, status=res_status))

# ---------------------------------------------------
# ROUTES: PREDICTION
# ---------------------------------------------------

@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    if not is_logged_in():
        return {"error": "Unauthorized"}, 401

    if not flare_up_model:
        return {"error": "Machine Learning model is offline"}, 500

    payload = request.json or {}

    try:
        with get_db() as conn:
            profile_row = conn.execute(
                "SELECT age, bmi, h_pylori_result, condition FROM users WHERE id = ?",
                (session["user_id"],)
            ).fetchone()

        if not profile_row:
            return {"error": "User profile not found"}, 404

        condition_value = (profile_row["condition"] or session.get("condition") or "general").lower()
        hpylori_value = (profile_row["h_pylori_result"] or "Not Applicable")
        if condition_value != "hpylori":
            hpylori_value = "Not Applicable"
        meal_type = payload.get("Meal_Type", "Lunch")
        food_name = normalize_food_name(payload.get("Food_Name"))
        meal_pral, pral_source = estimate_pral_score(food_name, meal_type=meal_type)
        symptom_context = get_recent_symptom_burden(session["user_id"])
        user_data = {
            "Age": to_float(profile_row["age"], 30),
            "BMI": to_float(profile_row["bmi"], 24.5),
            "Primary_Condition": MODEL_CONDITION_MAP.get(condition_value, "General"),
            "H_Pylori_Result": hpylori_value,
            "Food_Name": food_name,
            "Meal_Type": meal_type,
            "Meal_PRAL_Score": to_float(meal_pral, 0),
            "Water_Consumed_ml": int(to_float(payload.get("Water_Consumed_ml"), 200)),
            "Stress_At_Meal": payload.get("Stress_At_Meal", "Moderate")
        }

        # 2. Convert the single user entry into a Pandas DataFrame
        input_df = pd.DataFrame([user_data])

        # 3. Process the text categories into 1s and 0s (One-Hot Encoding)
        input_encoded = pd.get_dummies(input_df)

        # 4. Make sure the columns match exactly what the model learned during training
        # If the user didn't submit a category the model expects, fill it with a 0
        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder the columns to match the exact training order
        input_encoded = input_encoded[model_columns]

        # 5. Ask the model to predict the Flare-Up Score!
        prediction = flare_up_model.predict(input_encoded)[0]
        symptom_adjustment = 0.0
        if symptom_context["burden_level"] == "high":
            symptom_adjustment = 0.55
        elif symptom_context["burden_level"] == "moderate":
            symptom_adjustment = 0.25

        if symptom_context["trend_delta"] >= 3:
            symptom_adjustment += 0.2

        adjusted_prediction = max(0.0, min(10.0, float(prediction) + symptom_adjustment))
        result = build_prediction_result(adjusted_prediction)
        quality_info = assess_prediction_quality(food_name, pral_source, model_columns)

        if quality_info["is_uncertain"]:
            result["level"] = "uncertain"
            result["title"] = "Uncertain Result"
            result["message"] = "This food is not well-represented in known data, so this prediction may be unreliable."
            result["recommendation"] = "Needs Review"
            result["recommendation_reason"] = "Not enough trusted food-specific data for a reliable recommendation."
            result["confidence_label"] = "Low"
            result["confidence_percent"] = min(int(result.get("confidence_percent", 50)), 55)

        result["why_points"] = build_prediction_why_points(
            user_data=user_data,
            condition_value=condition_value,
            food_name=food_name,
            looked_up_pral=meal_pral if pral_source == "knowledge_base" else None
        )
        if symptom_context["recent_count"] > 0:
            result["why_points"].insert(
                1,
                f"Recent symptom burden is {symptom_context['burden_level']} ({symptom_context['recent_count']} logs in 7 days)."
            )
        result["why_points"].insert(0, quality_info["reason"])
        result["why_points"] = result["why_points"][:5]
        result["pral_score"] = round(float(meal_pral), 2)
        result["pral_source"] = pral_source
        result["prediction_quality"] = quality_info["quality"]
        result["symptom_burden"] = symptom_context
        result["raw_model_score"] = round(float(prediction), 2)
        result["symptom_adjustment"] = round(float(symptom_adjustment), 2)
        result["uncertainty_reason"] = quality_info["reason"] if quality_info["is_uncertain"] else None
        if result["level"] == "high":
            result["alternatives"] = suggest_alternative_meals(
                condition_value=condition_value,
                meal_type=meal_type,
                exclude_food=food_name,
                limit=3,
            )
        else:
            result["alternatives"] = []

        # Keep the latest prediction available to the template.
        session["last_prediction"] = result
        session["last_prediction_context"] = {
            "food_name": food_name,
            "risk_level": result["level"],
            "why_points": result.get("why_points", []),
            "alternatives": result.get("alternatives", []),
            "recommendation": result.get("recommendation", "Use Caution"),
            "condition": condition_value,
        }

        # 6. Return the score and interpretation back to the website
        return {
            "flare_up_risk_score": result["score"],
            "risk_level": result["level"],
            "title": result["title"],
            "message": result["message"],
            "recommendation": result["recommendation"],
            "recommendation_reason": result["recommendation_reason"],
            "why_points": result["why_points"],
            "pral_score": result["pral_score"],
            "pral_source": result["pral_source"],
            "prediction_quality": result["prediction_quality"],
            "alternatives": result["alternatives"],
            "confidence_percent": result["confidence_percent"],
            "confidence_label": result["confidence_label"],
            "calibration_hint": result.get("calibration_hint"),
            "decision_threshold": result.get("decision_threshold"),
            "high_risk_threshold": result.get("high_risk_threshold"),
            "raw_model_score": result.get("raw_model_score"),
            "symptom_adjustment": result.get("symptom_adjustment"),
            "symptom_burden": result.get("symptom_burden"),
            "uncertainty_reason": result.get("uncertainty_reason"),
            "generated_at": result["generated_at"]
        }

    except Exception as e:
        return {"error": str(e)}, 400


@app.route('/meal_followup_chat', methods=['POST'])
def meal_followup_chat():
    if not is_logged_in():
        return {"error": "Unauthorized"}, 401

    context = session.get("last_prediction_context")
    if not context:
        return {"error": "No meal has been analyzed yet. Predict a meal first."}, 400

    payload = request.json or {}
    question = (payload.get("question") or "").strip()
    answer = build_followup_answer(question, context)
    return {
        "answer": answer,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

# ---------------------------------------------------
# ROUTES: SYMPTOMS & MEDS
# ---------------------------------------------------
@app.route("/view_symptoms")
def view_symptoms():
    if not is_logged_in():
        return redirect(url_for("login"))

    severity_map = {"mild": 1.0, "moderate": 2.0, "severe": 3.0}

    with get_db() as conn:
        uid = session["user_id"]
        logs = conn.execute(
            "SELECT id, symptom, severity, note, timestamp FROM symptoms WHERE user_id = ? ORDER BY timestamp DESC LIMIT 60",
            (uid,)
        ).fetchall()

        recent_summary = conn.execute(
            """
            SELECT
                SUM(CASE WHEN timestamp >= datetime('now', '-7 day') THEN 1 ELSE 0 END) AS recent_count,
                SUM(CASE WHEN timestamp >= datetime('now', '-14 day') AND timestamp < datetime('now', '-7 day') THEN 1 ELSE 0 END) AS previous_count,
                SUM(CASE WHEN timestamp >= datetime('now', '-7 day') AND LOWER(severity) = 'severe' THEN 1 ELSE 0 END) AS severe_recent
            FROM symptoms
            WHERE user_id = ?
            """,
            (uid,)
        ).fetchone()

        top_symptom = conn.execute(
            """
            SELECT symptom, COUNT(*) AS cnt
            FROM symptoms
            WHERE user_id = ?
            GROUP BY LOWER(symptom)
            ORDER BY cnt DESC, LOWER(symptom) ASC
            LIMIT 1
            """,
            (uid,)
        ).fetchone()

        daily_rows = conn.execute(
            """
            SELECT date(timestamp) AS day, COUNT(*) AS cnt
            FROM symptoms
            WHERE user_id = ? AND timestamp >= datetime('now', '-13 day')
            GROUP BY date(timestamp)
            ORDER BY day ASC
            """,
            (uid,)
        ).fetchall()

        trigger_rows = conn.execute(
            """
            SELECT
                s.symptom AS symptom,
                f.food_name AS food_name,
                COUNT(*) AS hit_count,
                AVG(CASE LOWER(s.severity)
                    WHEN 'mild' THEN 1
                    WHEN 'moderate' THEN 2
                    WHEN 'severe' THEN 3
                    ELSE 1
                END) AS avg_severity
            FROM symptoms s
            JOIN food_logs f
                ON s.user_id = f.user_id
               AND julianday(s.timestamp) - julianday(f.timestamp) BETWEEN 0 AND 0.25
            WHERE s.user_id = ?
            GROUP BY LOWER(s.symptom), LOWER(f.food_name)
            HAVING hit_count >= 2
            ORDER BY hit_count DESC, avg_severity DESC, LOWER(f.food_name) ASC
            LIMIT 6
            """,
            (uid,)
        ).fetchall()

    daily_map = {row["day"]: int(row["cnt"] or 0) for row in daily_rows}
    day_labels = []
    day_counts = []
    for delta in range(13, -1, -1):
        day = (date.today() - timedelta(days=delta)).isoformat()
        day_labels.append(day[5:])
        day_counts.append(daily_map.get(day, 0))

    recent_count = int((recent_summary["recent_count"] or 0) if recent_summary else 0)
    previous_count = int((recent_summary["previous_count"] or 0) if recent_summary else 0)
    severe_recent = int((recent_summary["severe_recent"] or 0) if recent_summary else 0)

    recent_logs = [row for row in logs if row["timestamp"] and row["timestamp"] >= (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")]
    avg_severity_7d = 0.0
    if recent_logs:
        avg_severity_7d = round(
            sum(severity_map.get((row["severity"] or "mild").strip().lower(), 1.0) for row in recent_logs) / len(recent_logs),
            2
        )

    trend_delta = recent_count - previous_count
    if trend_delta >= 3:
        trend_label = "Worsening"
    elif trend_delta <= -3:
        trend_label = "Improving"
    else:
        trend_label = "Stable"

    trigger_insights = []
    for row in trigger_rows:
        avg_sev = round(float(row["avg_severity"] or 0), 2)
        confidence = "high" if row["hit_count"] >= 4 else "medium"
        if avg_sev >= 2.5:
            confidence = "high"
        trigger_insights.append({
            "symptom": row["symptom"],
            "food_name": row["food_name"],
            "hit_count": int(row["hit_count"] or 0),
            "avg_severity": avg_sev,
            "confidence": confidence,
        })

    action_items = []
    if severe_recent >= 2:
        action_items.append("You logged multiple severe symptoms this week. Consider contacting your clinician if this pattern continues.")
    if trend_delta >= 3:
        action_items.append("Symptoms rose versus last week. Review recent meals and prioritize lower-acid, lower-fat options.")
    if trigger_insights:
        top_trigger = trigger_insights[0]
        action_items.append(
            f"{top_trigger['food_name'].title()} appears linked to {top_trigger['symptom'].lower()} ({top_trigger['hit_count']} close-time occurrences). Trial reducing it for 7 days."
        )
    if not action_items:
        action_items.append("Current symptom pattern is relatively steady. Keep logging daily so trigger detection becomes more accurate.")

    summary = {
        "recent_count": recent_count,
        "previous_count": previous_count,
        "severe_recent": severe_recent,
        "avg_severity_7d": avg_severity_7d,
        "trend_delta": trend_delta,
        "trend_label": trend_label,
        "top_symptom": top_symptom["symptom"] if top_symptom else None,
    }

    return render_template(
        "view_symptoms.html",
        logs=logs,
        summary=summary,
        trend_labels=day_labels,
        trend_counts=day_counts,
        trigger_insights=trigger_insights,
        action_items=action_items,
    )

@app.route("/add_symptom", methods=["POST"])
def add_symptom():
    if not is_logged_in():
        return redirect(url_for("login"))
    with get_db() as conn:
        conn.execute("INSERT INTO symptoms (user_id, symptom, severity, note) VALUES (?, ?, ?, ?)",
                     (session["user_id"], request.form["symptom"], request.form.get("severity", "Mild"), request.form.get("note", "")))
    return redirect(url_for("view_symptoms"))

@app.route("/view_meds")
def view_meds():
    if not is_logged_in():
        return redirect(url_for("login"))
    with get_db() as conn:
        meds = conn.execute("SELECT * FROM medications WHERE user_id = ? ORDER BY id DESC", (session["user_id"],)).fetchall()
    return render_template("view_meds.html", meds=meds)

@app.route("/add_med", methods=["POST"])
def add_med():
    if not is_logged_in():
        return redirect(url_for("login"))
    # Join multiple time inputs into a single string for storage
    times_str = ", ".join(request.form.getlist("time"))
    with get_db() as conn:
        conn.execute("INSERT INTO medications (user_id, name, dosage, time, times_per_day, duration_days, instructions) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (session["user_id"], request.form["name"], request.form.get("dosage"), times_str, 
                      request.form.get("times_per_day", 1), request.form.get("duration_days", 14), request.form.get("instructions", "")))
    flash("Medication added.")
    return redirect(url_for("view_meds"))

@app.route("/mark_taken/<int:med_id>", methods=["POST"])
def mark_taken(med_id):
    if not is_logged_in():
        return redirect(url_for("login"))
    with get_db() as conn:
        med = conn.execute("SELECT * FROM medications WHERE id = ? AND user_id = ?", (med_id, session["user_id"])).fetchone()
        if not med:
            flash("Medication not found.")
            return redirect(url_for("view_meds"))

        # Prevent logging more than allowed per day
        today = date.today().isoformat()
        # Reset taken_today if last_taken_date is not today
        if med["last_taken_date"] != today:
            conn.execute("UPDATE medications SET taken_today = 0, last_taken_date = ? WHERE id = ? AND user_id = ?", (today, med_id, session["user_id"]))
            med = dict(med)
            med["taken_today"] = 0
            med["last_taken_date"] = today

        if med["taken_today"] >= med["times_per_day"]:
            flash(f"You have already logged all {med['times_per_day']} doses for today.")
            return redirect(url_for("view_meds"))

        # Parse scheduled times (may be multiple, comma-separated)
        scheduled_times = [t.strip() for t in (med["time"] or "").split(",") if t.strip()]
        now_dt = datetime.now()
        now_time = now_dt.time()
        allowed = False
        window_msg = None
        for idx, t in enumerate(scheduled_times):
            try:
                sched_dt = datetime.combine(now_dt.date(), datetime.strptime(t, "%H:%M").time())
                window_start = (sched_dt - timedelta(minutes=30)).time()
                window_end = (sched_dt + timedelta(hours=1)).time()
                # Only allow logging if now is within the window for the next unlogged dose
                # Count how many doses have been logged today
                if med["taken_today"] == idx:
                    if window_start <= now_time <= window_end:
                        allowed = True
                        break
                    else:
                        window_msg = f"You can only log this dose between {window_start.strftime('%H:%M')} and {window_end.strftime('%H:%M')}."
                        break
            except Exception:
                continue
        if not allowed:
            flash(window_msg or "You cannot log this dose at this time.")
            return redirect(url_for("view_meds"))

        # Increment both today's and total taken
        conn.execute("UPDATE medications SET taken_today = taken_today + 1, total_taken = total_taken + 1, last_taken_date = ? WHERE id = ? AND user_id = ?",
                     (today, med_id, session["user_id"]))
    flash("Dose recorded.")
    return redirect(url_for("view_meds"))

@app.route("/insights")
def insights():
    if not is_logged_in(): return redirect(url_for("login"))
    with get_db() as conn:
        uid = session["user_id"]
        frequency = conn.execute(
            "SELECT symptom, COUNT(*) AS count FROM symptoms WHERE user_id = ? GROUP BY LOWER(symptom) ORDER BY count DESC, LOWER(symptom) ASC LIMIT 10",
            (uid,)
        ).fetchall()
        correlations = conn.execute(
            """
            SELECT s.symptom, f.food_name
            FROM symptoms s
            JOIN food_logs f
              ON s.user_id = f.user_id
             AND julianday(s.timestamp) - julianday(f.timestamp) BETWEEN 0 AND 0.1667
            WHERE s.user_id = ?
            ORDER BY s.timestamp DESC
            LIMIT 8
            """,
            (uid,)
        ).fetchall()
    return render_template("insights.html", frequency=frequency, correlations=correlations)

if __name__ == "__main__":
    init_db()
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)