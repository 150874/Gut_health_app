
from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import date, datetime
from contextlib import contextmanager
import logging

# ---------------------------------------------------
# APP SETUP
# ---------------------------------------------------
app = Flask(__name__)
app.secret_key = "gut-health-secret-key"

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

# ---------------------------------------------------
# INITIALIZATION & SEEDING
# ---------------------------------------------------
def init_db():
    with get_db() as conn:
        c = conn.cursor()
        # Create all tables (Users, Food Rules, Logs, Symptoms, Knowledge, Medications)
        c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT UNIQUE, password TEXT, condition TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS food_rules (id INTEGER PRIMARY KEY AUTOINCREMENT, food_name TEXT UNIQUE, gastritis_status TEXT, gerd_status TEXT, hpylori_status TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS food_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, food_name TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        c.execute("CREATE TABLE IF NOT EXISTS symptoms (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, symptom TEXT, severity TEXT, note TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        c.execute("CREATE TABLE IF NOT EXISTS symptom_knowledge (id INTEGER PRIMARY KEY AUTOINCREMENT, symptom_name TEXT UNIQUE, duration TEXT, seek_help TEXT, tips TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS medications (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, name TEXT, dosage TEXT, time TEXT, times_per_day INTEGER DEFAULT 1, duration_days INTEGER DEFAULT 14, instructions TEXT, taken_today INTEGER DEFAULT 0, last_taken_date TEXT, total_taken INTEGER DEFAULT 0)")

            # --- Migration: Add total_taken column if missing ---
        c.execute("PRAGMA table_info(medications)")
        columns = [row[1] for row in c.fetchall()]
        if 'total_taken' not in columns:
                c.execute("ALTER TABLE medications ADD COLUMN total_taken INTEGER DEFAULT 0")

        # Seed data if empty
        c.execute("SELECT COUNT(*) FROM food_rules")
        if c.fetchone()[0] == 0:
            foods = [("Coffee", "avoid", "avoid", "avoid"), ("Banana", "safe", "safe", "safe"), ("Lemon", "safe", "avoid", "neutral"), ("Oatmeal", "safe", "safe", "safe")]
            c.executemany("INSERT INTO food_rules (food_name, gastritis_status, gerd_status, hpylori_status) VALUES (?, ?, ?, ?)", foods)

def is_logged_in():
    return "user_id" in session

# ---------------------------------------------------
# ROUTES: DASHBOARD & AUTH
# ---------------------------------------------------
@app.route("/")
def home():
    if not is_logged_in():
        return redirect(url_for("login"))

    uid = session["user_id"]
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
                           current_date=display_date, 
                           meals=meals, 
                           meds_count=taken_doses, 
                           symptoms_count=symptoms_count, 
                           progress=progress,
                           meds=meds)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        hashed_pw = generate_password_hash(request.form["password"])
        try:
            with get_db() as conn:
                conn.execute("INSERT INTO users (name, email, password, condition) VALUES (?, ?, ?, ?)",
                             (request.form["name"], request.form["email"], hashed_pw, request.form["condition"]))
            flash("Account created!")
            return redirect(url_for("login"))
        except:
            flash("Email already exists.")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (request.form["email"],)).fetchone()
        if user and check_password_hash(user["password"], request.form["password"]):
            session.update({"user_id": user["id"], "user_name": user["name"], "condition": user["condition"]})
            return redirect(url_for("home"))
        flash("Invalid credentials.")
    return render_template("login.html")

@app.route("/logout") # Matches url_for in the new templates
def logout_page():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("login"))

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
        recent_logs = conn.execute("SELECT food_name, timestamp FROM food_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5", (session["user_id"],)).fetchall()
        
    return render_template("diet_checker.html", 
                           food=food, 
                           status=status, 
                           recent_logs=recent_logs, 
                           condition=session["condition"])

@app.route("/check", methods=["POST"])
def check():
    if not is_logged_in():
        return redirect(url_for("login"))

    food_query = request.form["food"].strip().lower()
    condition = session["condition"]
    column = CONDITION_COLUMN_MAP.get(condition, "gastritis_status")

    with get_db() as conn:
        row = conn.execute(f"SELECT food_name, {column} AS status FROM food_rules WHERE LOWER(food_name) = ?", (food_query,)).fetchone()
        
        if row:
            conn.execute("INSERT INTO food_logs (user_id, food_name) VALUES (?, ?)", (session["user_id"], row["food_name"]))
            res_food, res_status = row["food_name"], row["status"]
        else:
            res_food, res_status = food_query, "unknown"

    # Redirect to the diet_checker page with result data in the URL
    return redirect(url_for("diet_checker", food=res_food, status=res_status))

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
    app.run(debug=True)