from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import sqlite3
from datetime import datetime, date
from contextlib import contextmanager
import logging

app = Flask(__name__)
app.secret_key = "bbit-dev-secret-key" # Security note: Use env vars in production

logging.basicConfig(level=logging.INFO)
DB_PATH = "database.db"

CONDITION_COLUMN_MAP = {
    "gastritis": "gastritis_status",
    "gerd": "gerd_status",
    "hpylori": "hpylori_status"
}

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

# -----------------------------
# DATABASE INITIALIZATION
# -----------------------------
def init_db():
    with get_db() as conn:
        c = conn.cursor()

        # User Profile for personalization
        c.execute('''CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            name TEXT DEFAULT 'Sarah',
            condition TEXT DEFAULT 'gastritis'
        )''')
        c.execute("INSERT OR IGNORE INTO user_profile (id) VALUES (1)")

        # Food Rules and Logs
        c.execute('''CREATE TABLE IF NOT EXISTS food_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_name TEXT UNIQUE,
            gastritis_status TEXT,
            gerd_status TEXT,
            hpylori_status TEXT
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS food_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')

        # Symptoms and Knowledge Base
        c.execute('''CREATE TABLE IF NOT EXISTS symptoms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptom TEXT,
            severity TEXT, -- Mild, Moderate, Severe
            note TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS symptom_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptom_name TEXT UNIQUE,
            duration TEXT,
            seek_help TEXT,
            tips TEXT -- Stored as semi-colon separated string
        )''')

        # Medications (Upgraded for Progress Tracking)
        c.execute('''CREATE TABLE IF NOT EXISTS medications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dosage TEXT,
            time TEXT NOT NULL,
            times_per_day INTEGER DEFAULT 1,
            duration_days INTEGER DEFAULT 14,
            instructions TEXT,
            taken_today INTEGER DEFAULT 0, -- Resets daily
            last_taken_date TEXT
        )''')

        # Seed educational content (Ref: image_87b34c.png)
        c.execute("SELECT COUNT(*) FROM symptom_knowledge")
        if c.fetchone()[0] == 0:
            c.execute('''INSERT INTO symptom_knowledge (symptom_name, duration, seek_help, tips) VALUES 
                (?, ?, ?, ?)''', (
                    "Bloating", 
                    "Usually lasts 30 minutes to a few hours", 
                    "Contact your doctor if bloating is severe, persistent (lasting more than a week), or accompanied by abdominal pain.",
                    "Eat slowly and chew food thoroughly; Limit foods known to cause gas; Drink plenty of water; Light walking"
                ))

# -----------------------------
# DASHBOARD (image_87b7a7.png)
# -----------------------------
@app.route("/")
def home():
    today = date.today().isoformat()
    with get_db() as conn:
        # Get counts for Today's Summary
        meals_count = conn.execute("SELECT COUNT(*) FROM food_logs WHERE date(timestamp) = ?", (today,)).fetchone()[0]
        symptoms_count = conn.execute("SELECT COUNT(*) FROM symptoms WHERE date(timestamp) = ?", (today,)).fetchone()[0]
        
        # Medication Progress
        meds = conn.execute("SELECT name, taken_today, times_per_day FROM medications").fetchall()
        total_doses = sum(m['times_per_day'] for m in meds)
        taken_doses = sum(m['taken_today'] for m in meds)
        progress = int((taken_doses / total_doses * 100)) if total_doses > 0 else 0

        user = conn.execute("SELECT * FROM user_profile WHERE id = 1").fetchone()

    return render_template("index.html", 
                           user=user, 
                           meals=meals_count, 
                           meds_count=taken_doses, 
                           symptoms_count=symptoms_count,
                           progress=progress)

# -----------------------------
# SYMPTOM TRACKER (image_87b3e5.png, image_87b34c.png)
# -----------------------------
@app.route("/view_symptoms")
def view_symptoms():
    with get_db() as conn:
        logs = conn.execute("SELECT * FROM symptoms ORDER BY timestamp DESC").fetchall()
        # Fetch knowledge for the most recent symptom to show "Insights"
        recent_symptom = logs[0]['symptom'] if logs else None
        knowledge = conn.execute("SELECT * FROM symptom_knowledge WHERE symptom_name = ?", (recent_symptom,)).fetchone()
        
    tips = knowledge['tips'].split(';') if knowledge else []
    return render_template("view_symptoms.html", logs=logs, knowledge=knowledge, tips=tips)

@app.route("/add_symptom", methods=["POST"])
def add_symptom():
    symptom = request.form["symptom"]
    severity = request.form.get("severity", "Mild")
    note = request.form.get("note", "")
    with get_db() as conn:
        conn.execute("INSERT INTO symptoms (symptom, severity, note) VALUES (?, ?, ?)", (symptom, severity, note))
    return redirect(url_for("view_symptoms"))

# -----------------------------
# MEDICATION TRACKER (image_87abb2.png)
# -----------------------------
@app.route("/view_meds")
def view_meds():
    with get_db() as conn:
        meds = conn.execute("SELECT * FROM medications ORDER BY time ASC").fetchall()
    return render_template("view_meds.html", meds=meds)

@app.route("/mark_taken/<int:med_id>", methods=["POST"])
def mark_taken(med_id):
    today = date.today().isoformat()
    with get_db() as conn:
        conn.execute('''UPDATE medications 
                        SET taken_today = taken_today + 1, 
                            last_taken_date = ? 
                        WHERE id = ?''', (today, med_id))
    return redirect(url_for("view_meds"))

# -----------------------------
# DIET CHECKER (image_87b026.png)
# -----------------------------
@app.route("/check", methods=["POST"])
def check():
    food = request.form["food"].strip()
    condition = request.form.get("condition", "gastritis")
    col = CONDITION_COLUMN_MAP.get(condition, "gastritis_status")

    with get_db() as conn:
        row = conn.execute(f"SELECT {col} FROM food_rules WHERE LOWER(food_name) = LOWER(?)", (food,)).fetchone()
        # Log the meal if checked and safe (optional logic)
        if row and row[0] == 'safe':
            conn.execute("INSERT INTO food_logs (food_name) VALUES (?)", (food,))

    status = row[0] if row else "unknown"
    return render_template("index.html", food=food, status=status)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)