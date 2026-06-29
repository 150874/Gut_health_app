
from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import date, datetime, timedelta
from contextlib import contextmanager
import logging
import joblib
import pandas as pd
import json
import os
import secrets


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


def build_prediction_result(score):
    raw_score = float(score)
    score = round(raw_score, 1)
    bounded_score = max(0.0, min(10.0, raw_score))

    # Confidence is higher when the score is farther from decision boundaries (4 and 7).
    distance_to_boundary = min(abs(bounded_score - 4.0), abs(bounded_score - 7.0))
    boundary_confidence = min(distance_to_boundary / 3.0, 1.0)
    confidence_score = 0.55 + (0.40 * boundary_confidence)

    # Calibrate by recent model quality when available.
    quality_signal = None
    if isinstance(model_test_results, dict):
        quality_signal = model_test_results.get("balanced_accuracy")
    if quality_signal is None and isinstance(model_test_results, dict):
        quality_signal = model_test_results.get("f1")
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

    if score >= 7:
        level = "high"
        title = "DANGER!"
        message = "High probability of a flare-up. Consider drinking more water or changing the meal."
    elif score >= 4:
        level = "moderate"
        title = "Warning."
        message = "Moderate risk. Proceed with caution."
    else:
        level = "low"
        title = "All Clear!"
        message = "This meal profile looks safe for your gut health."

    return {
        "score": score,
        "level": level,
        "title": title,
        "message": message,
        "confidence_percent": confidence_percent,
        "confidence_label": confidence_label,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

# ---------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
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


@app.route("/admin")
def admin_dashboard():
    guard = require_admin()
    if guard:
        return guard

    training_history = load_training_history()
    recent_training_history = list(reversed(training_history[-10:]))
    feature_importance = load_feature_importance()

    with get_db() as conn:
        admin_users = conn.execute(
            "SELECT id, name, email, condition, is_admin, is_active FROM users ORDER BY is_admin DESC, id DESC"
        ).fetchall()
        pral_entries = conn.execute(
            "SELECT id, food_name, pral_score, source, updated_at FROM food_pral ORDER BY LOWER(food_name) ASC"
        ).fetchall()
        audit_logs = conn.execute(
            "SELECT a.created_at, a.action_type, a.target, a.details, u.name AS admin_name "
            "FROM admin_audit_logs a LEFT JOIN users u ON u.id = a.admin_user_id "
            "ORDER BY a.id DESC LIMIT 25"
        ).fetchall()

    return render_template(
        "admin.html",
        model_test_results=model_test_results,
        admin_users=admin_users,
        training_history=recent_training_history,
        pral_entries=pral_entries,
        feature_importance=feature_importance,
        audit_logs=audit_logs
    )


@app.route("/admin/pral/add", methods=["POST"])
def admin_add_pral():
    guard = require_admin()
    if guard:
        return guard

    food_name = request.form.get("food_name", "").strip()
    pral_score = to_float(request.form.get("pral_score"), 0)
    if not food_name:
        flash("Food name is required.")
        return redirect(url_for("admin_dashboard"))

    with get_db() as conn:
        conn.execute(
            "INSERT INTO food_pral (food_name, pral_score, source, updated_at) VALUES (?, ?, 'manual', CURRENT_TIMESTAMP) "
            "ON CONFLICT(food_name) DO UPDATE SET pral_score = excluded.pral_score, source = 'manual', updated_at = CURRENT_TIMESTAMP",
            (food_name, pral_score)
        )
    log_admin_action("pral_upsert", food_name, f"pral_score={pral_score}")
    flash("PRAL entry saved.")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/pral/<int:entry_id>/delete", methods=["POST"])
def admin_delete_pral(entry_id):
    guard = require_admin()
    if guard:
        return guard

    with get_db() as conn:
        target = conn.execute("SELECT food_name, pral_score FROM food_pral WHERE id = ?", (entry_id,)).fetchone()
        conn.execute("DELETE FROM food_pral WHERE id = ?", (entry_id,))
    if target:
        log_admin_action("pral_delete", target["food_name"], f"pral_score={target['pral_score']}")
    flash("PRAL entry deleted.")
    return redirect(url_for("admin_dashboard"))


@app.route("/api/pral-lookup")
def pral_lookup_api():
    if not is_logged_in():
        return {"error": "Unauthorized"}, 401
    food_name = request.args.get("food", "").strip()
    if not food_name:
        return {"found": False, "pral_score": None}
    score = lookup_pral_score(food_name)
    if score is None:
        return {"found": False, "pral_score": None}
    return {"found": True, "pral_score": round(float(score), 2)}


@app.route("/admin/users/<int:user_id>/toggle-admin", methods=["POST"])
def admin_toggle_user_role(user_id):
    guard = require_admin()
    if guard:
        return guard

    current_admin_id = session["user_id"]
    with get_db() as conn:
        target = conn.execute("SELECT id, name, email, is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
        if not target:
            flash("User not found.")
            return redirect(url_for("admin_dashboard"))

        if target["id"] == current_admin_id and target["is_admin"]:
            flash("You cannot remove your own admin role while logged in.")
            return redirect(url_for("admin_dashboard"))

        new_role = 0 if target["is_admin"] else 1
        conn.execute("UPDATE users SET is_admin = ? WHERE id = ?", (new_role, user_id))
    action = "user_promote" if new_role == 1 else "user_demote"
    log_admin_action(action, target["email"], f"target_name={target['name']}")

    flash("User role updated.")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users/<int:user_id>/toggle-active", methods=["POST"])
def admin_toggle_user_active(user_id):
    guard = require_admin()
    if guard:
        return guard

    current_admin_id = session["user_id"]
    with get_db() as conn:
        target = conn.execute("SELECT id, name, email, is_active FROM users WHERE id = ?", (user_id,)).fetchone()
        if not target:
            flash("User not found.")
            return redirect(url_for("admin_dashboard"))

        if target["id"] == current_admin_id and target["is_active"]:
            flash("You cannot deactivate your own account while logged in.")
            return redirect(url_for("admin_dashboard"))

        new_status = 0 if target["is_active"] else 1
        conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (new_status, user_id))
    action = "user_activate" if new_status == 1 else "user_deactivate"
    log_admin_action(action, target["email"], f"target_name={target['name']}")

    flash("User status updated.")
    return redirect(url_for("admin_dashboard"))

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
        meal_pral = payload.get("Meal_PRAL_Score")
        food_name = normalize_food_name(payload.get("Food_Name"))
        if meal_pral in (None, ""):
            looked_up_pral = lookup_pral_score(food_name)
            meal_pral = looked_up_pral if looked_up_pral is not None else 0
        user_data = {
            "Age": to_float(profile_row["age"], 30),
            "BMI": to_float(profile_row["bmi"], 24.5),
            "Primary_Condition": MODEL_CONDITION_MAP.get(condition_value, "General"),
            "H_Pylori_Result": hpylori_value,
            "Food_Name": food_name,
            "Meal_Type": payload.get("Meal_Type", "Lunch"),
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
        result = build_prediction_result(prediction)

        # Keep the latest prediction available to the template.
        session["last_prediction"] = result

        # 6. Return the score and interpretation back to the website
        return {
            "flare_up_risk_score": result["score"],
            "risk_level": result["level"],
            "title": result["title"],
            "message": result["message"],
            "confidence_percent": result["confidence_percent"],
            "confidence_label": result["confidence_label"],
            "generated_at": result["generated_at"]
        }

    except Exception as e:
        return {"error": str(e)}, 400

# ---------------------------------------------------
# ROUTES: SYMPTOMS & MEDS
# ---------------------------------------------------
@app.route("/view_symptoms")
def view_symptoms():
    if not is_logged_in():
        return redirect(url_for("login"))
    with get_db() as conn:
        logs = conn.execute("SELECT * FROM symptoms WHERE user_id = ? ORDER BY timestamp DESC", (session["user_id"],)).fetchall()
    return render_template("view_symptoms.html", logs=logs)

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
        top_symptoms = conn.execute("SELECT symptom, COUNT(*) as count FROM symptoms WHERE user_id = ? GROUP BY symptom ORDER BY count DESC", (session["user_id"],)).fetchall()
        top_foods = conn.execute("SELECT food_name, COUNT(*) as count FROM food_logs WHERE user_id = ? GROUP BY food_name ORDER BY count DESC", (session["user_id"],)).fetchall()
    return render_template("insights.html", top_symptoms=top_symptoms, top_foods=top_foods)

if __name__ == "__main__":
    init_db()
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)