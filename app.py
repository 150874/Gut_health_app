from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
from contextlib import contextmanager
import logging

# ---------------------------------------------------
# APP SETUP
# ---------------------------------------------------
app = Flask(__name__)
app.secret_key = "gut-health-secret-key"

logging.basicConfig(level=logging.INFO)
DB_PATH = "database.db"

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
        logging.error(f"Database error: {e}")
        raise
    finally:
        conn.close()

# ---------------------------------------------------
# DATABASE INITIALIZATION
# ---------------------------------------------------
def init_db():
    with get_db() as conn:
        c = conn.cursor()
        
        # USERS TABLE
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            condition TEXT NOT NULL
        )''')
        
        # FOOD RULES TABLE (Normalized)
        c.execute('''CREATE TABLE IF NOT EXISTS food_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_name TEXT UNIQUE,
            gastritis_status TEXT,
            gerd_status TEXT,
            hpylori_status TEXT
        )''')
        
        # FOOD LOGS TABLE (Linked to user_id)
        c.execute('''CREATE TABLE IF NOT EXISTS food_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            food_name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')
        
        # SYMPTOMS TABLE (Linked to user_id)
        c.execute('''CREATE TABLE IF NOT EXISTS symptoms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symptom TEXT,
            severity TEXT,
            note TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')
        
        # SYMPTOM KNOWLEDGE BASE
        c.execute('''CREATE TABLE IF NOT EXISTS symptom_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptom_name TEXT UNIQUE,
            duration TEXT,
            seek_help TEXT,
            tips TEXT
        )''')
        
        # MEDICATIONS TABLE (Linked to user_id)
        c.execute('''CREATE TABLE IF NOT EXISTS medications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            dosage TEXT,
            time TEXT NOT NULL,
            times_per_day INTEGER DEFAULT 1,
            duration_days INTEGER DEFAULT 14,
            instructions TEXT,
            taken_today INTEGER DEFAULT 0,
            last_taken_date TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')

        # SEED DATA: FOOD RULES
        c.execute("SELECT COUNT(*) FROM food_rules")
        if c.fetchone()[0] == 0:
            sample_foods = [
                ("Coffee", "avoid", "avoid", "avoid"),
                ("Banana", "safe", "safe", "safe"),
                ("Spicy Food", "avoid", "avoid", "avoid"),
                ("Oatmeal", "safe", "safe", "safe"),
                ("Tea", "neutral", "neutral", "safe"),
                ("Alcohol", "avoid", "avoid", "avoid"),
                ("White Rice", "safe", "safe", "safe"),
                ("Yogurt", "neutral", "safe", "safe"),
                ("Fried Food", "avoid", "avoid", "avoid"),
                ("Ginger Tea", "safe", "neutral", "safe")
            ]
            c.executemany('INSERT INTO food_rules (food_name, gastritis_status, gerd_status, hpylori_status) VALUES (?,?,?,?)', sample_foods)
        
        # SEED DATA: SYMPTOM KNOWLEDGE
        c.execute("SELECT COUNT(*) FROM symptom_knowledge")
        if c.fetchone()[0] == 0:
            c.execute('INSERT INTO symptom_knowledge (symptom_name, duration, seek_help, tips) VALUES (?,?,?,?)',
                      ("Bloating", 
                       "Usually lasts 30 minutes to a few hours", 
                       "Contact your doctor if bloating is severe or persistent.", 
                       "Eat slowly and chew food thoroughly; Limit gas-inducing foods; Drink water; Light walking; Avoid carbonated beverages"))

# ---------------------------------------------------
# AUTHENTICATION HELPERS
# ---------------------------------------------------
def login_required():
    return "user_id" in session

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------

@app.route("/")
def home():
    if not login_required(): return redirect(url_for("login"))
    today = date.today().isoformat()
    uid = session["user_id"]
    
    with get_db() as conn:
        # Aggregate dashboard summary [cite: 228, 364]
        meals = conn.execute("SELECT COUNT(*) FROM food_logs WHERE user_id=? AND date(timestamp)=?", (uid, today)).fetchone()[0]
        symptoms = conn.execute("SELECT COUNT(*) FROM symptoms WHERE user_id=? AND date(timestamp)=?", (uid, today)).fetchone()[0]
        
        # Medication progress tracking [cite: 229, 366]
        meds_data = conn.execute("SELECT times_per_day, taken_today FROM medications WHERE user_id=?", (uid,)).fetchall()
        total_doses = sum(m["times_per_day"] for m in meds_data)
        taken_doses = sum(m["taken_today"] for m in meds_data)
        progress = int((taken_doses / total_doses) * 100) if total_doses > 0 else 0
        
    return render_template("index.html", 
                           user_name=session["user_name"], 
                           meals=meals, 
                           meds_count=taken_doses, 
                           symptoms_count=symptoms, 
                           progress=progress)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email, password = request.form["email"], request.form["password"]
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password"], password):
            session.update({"user_id": user["id"], "user_name": user["name"], "condition": user["condition"]})
            return redirect(url_for("home"))
        flash("Invalid email or password.")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        hashed = generate_password_hash(request.form["password"])
        try:
            with get_db() as conn:
                conn.execute('INSERT INTO users (name, email, password, condition) VALUES (?,?,?,?)', 
                             (request.form["name"], request.form["email"], hashed, request.form["condition"]))
            flash("Account created successfully.")
            return redirect(url_for("login"))
        except: flash("Email already exists.")
    return render_template("signup.html")

@app.route("/logout", methods=["GET", "POST"])
def logout():
    """Combined logout route for confirmation and clearing session[cite: 346, 347]."""
    if request.method == "POST":
        session.clear()
        flash("You have been logged out.", "info")
        return redirect(url_for("login"))
    if not login_required(): return redirect(url_for("login"))
    return render_template("logout.html")

@app.route("/check", methods=["POST"])
def check():
    if not login_required(): return redirect(url_for("login"))
    food = request.form["food"].strip()
    col = CONDITION_COLUMN_MAP.get(session["condition"], "gastritis_status")
    
    with get_db() as conn:
        # Check rule and log meal simultaneously [cite: 321, 377]
        row = conn.execute(f"SELECT {col} FROM food_rules WHERE LOWER(food_name)=LOWER(?)", (food,)).fetchone()
        if row: conn.execute("INSERT INTO food_logs (user_id, food_name) VALUES (?,?)", (session["user_id"], food))
    
    return render_template("index.html", user_name=session["user_name"], food=food, status=row[0] if row else "unknown")

@app.route("/view_symptoms")
def view_symptoms():
    if not login_required(): return redirect(url_for("login"))
    with get_db() as conn:
        logs = conn.execute("SELECT * FROM symptoms WHERE user_id=? ORDER BY timestamp DESC", (session["user_id"],)).fetchall()
        # Fetch insights for the latest symptom [cite: 275, 323, 378]
        knowledge = conn.execute("SELECT * FROM symptom_knowledge WHERE symptom_name=?", (logs[0]["symptom"] if logs else None,)).fetchone()
    
    tips = knowledge["tips"].split(";") if knowledge else []
    return render_template("view_symptoms.html", logs=logs, knowledge=knowledge, tips=tips)

@app.route("/add_symptom", methods=["POST"])
def add_symptom():
    with get_db() as conn:
        conn.execute("INSERT INTO symptoms (user_id, symptom, severity, note) VALUES (?,?,?,?)", 
                     (session["user_id"], request.form["symptom"], request.form.get("severity", "Mild"), request.form.get("note", "")))
    return redirect(url_for("view_symptoms"))

@app.route("/view_meds")
def view_meds():
    if not login_required(): return redirect(url_for("login"))
    with get_db() as conn:
        meds = conn.execute("SELECT * FROM medications WHERE user_id=? ORDER BY time ASC", (session["user_id"],)).fetchall()
    return render_template("view_meds.html", meds=meds)

@app.route("/add_med", methods=["POST"])
def add_med():
    # Use getlist to capture all dynamic time inputs
    times_list = request.form.getlist("time")
    time_string = ", ".join(times_list) # Saves as "08:00, 14:00, 20:00"

    with get_db() as conn:
        conn.execute('''INSERT INTO medications 
            (user_id, name, dosage, time, times_per_day, duration_days, instructions) 
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (session["user_id"], 
             request.form["name"], 
             request.form.get("dosage"), 
             time_string, 
             request.form.get("times_per_day", 1), 
             request.form.get("duration_days", 14), 
             request.form.get("instructions", "")))
    return redirect(url_for("view_meds"))

@app.route("/mark_taken/<int:med_id>", methods=["POST"])
def mark_taken(med_id):
    with get_db() as conn:
        conn.execute("UPDATE medications SET taken_today = taken_today + 1, last_taken_date = ? WHERE id = ? AND user_id = ?", 
                     (date.today().isoformat(), med_id, session["user_id"]))
    return redirect(url_for("view_meds"))

@app.route("/insights")
def insights():
    if not login_required(): return redirect(url_for("login"))
    with get_db() as conn:
        top_symptoms = conn.execute("SELECT symptom, COUNT(*) AS count FROM symptoms WHERE user_id=? GROUP BY symptom ORDER BY count DESC", (session["user_id"],)).fetchall()
        top_foods = conn.execute("SELECT food_name, COUNT(*) AS count FROM food_logs WHERE user_id=? GROUP BY food_name ORDER BY count DESC", (session["user_id"],)).fetchall()
    return render_template("insights.html", top_symptoms=top_symptoms, top_foods=top_foods)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)