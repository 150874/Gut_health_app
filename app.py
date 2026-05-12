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

# ---------------------------------------------------
# CONDITION MAP
# ---------------------------------------------------
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

        # USERS
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                condition TEXT NOT NULL
            )
        ''')

        # FOOD RULES
        c.execute('''
            CREATE TABLE IF NOT EXISTS food_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                food_name TEXT UNIQUE,
                gastritis_status TEXT,
                gerd_status TEXT,
                hpylori_status TEXT
            )
        ''')

        # FOOD LOGS
        c.execute('''
            CREATE TABLE IF NOT EXISTS food_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                food_name TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # SYMPTOMS
        c.execute('''
            CREATE TABLE IF NOT EXISTS symptoms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symptom TEXT,
                severity TEXT,
                note TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # SYMPTOM KNOWLEDGE
        c.execute('''
            CREATE TABLE IF NOT EXISTS symptom_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symptom_name TEXT UNIQUE,
                duration TEXT,
                seek_help TEXT,
                tips TEXT
            )
        ''')

        # MEDICATIONS
        c.execute('''
            CREATE TABLE IF NOT EXISTS medications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                dosage TEXT,
                time TEXT NOT NULL,
                times_per_day INTEGER DEFAULT 1,
                duration_days INTEGER DEFAULT 14,
                instructions TEXT,
                meal_timing TEXT,
                reminder_enabled INTEGER DEFAULT 1,
                taken_today INTEGER DEFAULT 0,
                last_taken_date TEXT
            )
        ''')

        # SEED FOOD DATA
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

            c.executemany('''
                INSERT INTO food_rules
                (food_name, gastritis_status, gerd_status, hpylori_status)
                VALUES (?, ?, ?, ?)
            ''', sample_foods)

        # SEED SYMPTOM KNOWLEDGE
        c.execute("SELECT COUNT(*) FROM symptom_knowledge")

        if c.fetchone()[0] == 0:

            c.execute('''
                INSERT INTO symptom_knowledge
                (symptom_name, duration, seek_help, tips)
                VALUES (?, ?, ?, ?)
            ''', (

                "Bloating",

                "Usually lasts 30 minutes to a few hours",

                "Contact your doctor if bloating is severe or persistent.",

                "Drink water;Avoid spicy food;Walk lightly;Eat slowly"

            ))

# ---------------------------------------------------
# LOGIN REQUIRED
# ---------------------------------------------------
def login_required():

    if "user_id" not in session:
        return False

    return True

# ---------------------------------------------------
# HOME / DASHBOARD
# ---------------------------------------------------
@app.route("/")
def home():

    if not login_required():
        return redirect(url_for("login"))

    today = date.today().isoformat()

    with get_db() as conn:

        # MEALS TODAY
        meals_count = conn.execute(
            "SELECT COUNT(*) FROM food_logs WHERE date(timestamp) = ?",
            (today,)
        ).fetchone()[0]

        # SYMPTOMS TODAY
        symptoms_count = conn.execute(
            "SELECT COUNT(*) FROM symptoms WHERE date(timestamp) = ?",
            (today,)
        ).fetchone()[0]

        # MEDICATIONS
        meds = conn.execute(
            "SELECT * FROM medications ORDER BY time ASC"
        ).fetchall()

        total_doses = sum(m["times_per_day"] for m in meds)
        taken_doses = sum(m["taken_today"] for m in meds)

        progress = int(
            (taken_doses / total_doses) * 100
        ) if total_doses > 0 else 0

    return render_template(
        "index.html",
        name=session["user_name"],
        condition=session["condition"],
        meals=meals_count,
        symptoms_count=symptoms_count,
        meds_count=taken_doses,
        progress=progress,
        meds=meds
    )

# ---------------------------------------------------
# SIGN UP
# ---------------------------------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        condition = request.form["condition"]

        hashed_password = generate_password_hash(password)

        try:

            with get_db() as conn:

                conn.execute('''
                    INSERT INTO users
                    (name, email, password, condition)
                    VALUES (?, ?, ?, ?)
                ''', (

                    name,
                    email,
                    hashed_password,
                    condition

                ))

            flash("Account created successfully.")
            return redirect(url_for("login"))

        except:

            flash("Email already exists.")
            return redirect(url_for("signup"))

    return render_template("signup.html")

# ---------------------------------------------------
# LOGIN
# ---------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        with get_db() as conn:

            user = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            ).fetchone()

        if user and check_password_hash(user["password"], password):

            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["condition"] = user["condition"]

            flash("Login successful.")

            return redirect(url_for("home"))

        flash("Invalid email or password.")

    return render_template("login.html")


# ---------------------------------------------------
# LOGOUT CONFIRMATION PAGE
# ---------------------------------------------------
@app.route("/logout")
def logout_page():

    if not login_required():
        return redirect(url_for("login"))

    return render_template("logout.html")


# ---------------------------------------------------
# ACTUAL LOGOUT ACTION
# ---------------------------------------------------
@app.route("/confirm_logout", methods=["POST"])
def confirm_logout():

    session.clear()

    flash("You have been logged out successfully.", "info")

    return redirect(url_for("login"))

# ---------------------------------------------------
# FOOD CHECKER
# ---------------------------------------------------
@app.route("/check", methods=["POST"])
def check():

    if not login_required():
        return redirect(url_for("login"))

    food = request.form["food"].strip()

    condition = session["condition"]

    column = CONDITION_COLUMN_MAP.get(
        condition,
        "gastritis_status"
    )

    with get_db() as conn:

        row = conn.execute(f'''
            SELECT food_name, {column} AS status
            FROM food_rules
            WHERE LOWER(food_name) = LOWER(?)
        ''', (food,)).fetchone()

        if row:
            conn.execute(
                "INSERT INTO food_logs (food_name) VALUES (?)",
                (food,)
            )

    status = row["status"] if row else "unknown"

    return render_template(
        "index.html",
        name=session["user_name"],
        condition=condition,
        food=food,
        status=status
    )

# ---------------------------------------------------
# SYMPTOMS
# ---------------------------------------------------
@app.route("/view_symptoms")
def view_symptoms():

    if not login_required():
        return redirect(url_for("login"))

    with get_db() as conn:

        logs = conn.execute(
            "SELECT * FROM symptoms ORDER BY timestamp DESC"
        ).fetchall()

        recent_symptom = logs[0]["symptom"] if logs else None

        knowledge = conn.execute(
            "SELECT * FROM symptom_knowledge WHERE symptom_name = ?",
            (recent_symptom,)
        ).fetchone()

    tips = knowledge["tips"].split(";") if knowledge else []

    return render_template(
        "view_symptoms.html",
        logs=logs,
        knowledge=knowledge,
        tips=tips
    )

@app.route("/add_symptom", methods=["POST"])
def add_symptom():

    symptom = request.form["symptom"]

    severity = request.form.get(
        "severity",
        "Mild"
    )

    note = request.form.get(
        "note",
        ""
    )

    with get_db() as conn:

        conn.execute('''
            INSERT INTO symptoms
            (symptom, severity, note)
            VALUES (?, ?, ?)
        ''', (

            symptom,
            severity,
            note

        ))

    return redirect(url_for("view_symptoms"))

# ---------------------------------------------------
# MEDICATIONS
# ---------------------------------------------------
@app.route("/view_meds")
def view_meds():

    if not login_required():
        return redirect(url_for("login"))

    with get_db() as conn:

        meds = conn.execute(
            "SELECT * FROM medications ORDER BY time ASC"
        ).fetchall()

    return render_template(
        "view_meds.html",
        meds=meds
    )

@app.route("/add_med", methods=["POST"])
def add_med():

    name = request.form["name"]

    dosage = request.form.get("dosage")

    time = request.form["time"]

    times_per_day = request.form.get(
        "times_per_day",
        1
    )

    duration_days = request.form.get(
        "duration_days",
        14
    )

    instructions = request.form.get(
        "instructions",
        ""
    )

    meal_timing = request.form.get(
        "meal_timing",
        ""
    )

    with get_db() as conn:

        conn.execute('''
            INSERT INTO medications
            (
                name,
                dosage,
                time,
                times_per_day,
                duration_days,
                instructions,
                meal_timing
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (

            name,
            dosage,
            time,
            times_per_day,
            duration_days,
            instructions,
            meal_timing

        ))

    flash("Medication added.")

    return redirect(url_for("view_meds"))

@app.route("/mark_taken/<int:med_id>", methods=["POST"])
def mark_taken(med_id):

    today = date.today().isoformat()

    with get_db() as conn:

        conn.execute('''
            UPDATE medications
            SET taken_today = taken_today + 1,
                last_taken_date = ?
            WHERE id = ?
        ''', (

            today,
            med_id

        ))

    flash("Medication marked as taken.")

    return redirect(url_for("view_meds"))

# ---------------------------------------------------
# INSIGHTS
# ---------------------------------------------------
@app.route("/insights")
def insights():

    if not login_required():
        return redirect(url_for("login"))

    with get_db() as conn:

        top_symptoms = conn.execute('''
            SELECT symptom, COUNT(*) AS count
            FROM symptoms
            GROUP BY symptom
            ORDER BY count DESC
        ''').fetchall()

        top_foods = conn.execute('''
            SELECT food_name, COUNT(*) AS count
            FROM food_logs
            GROUP BY food_name
            ORDER BY count DESC
        ''').fetchall()

    return render_template(
        "insights.html",
        top_symptoms=top_symptoms,
        top_foods=top_foods
    )

# ---------------------------------------------------
# API
# ---------------------------------------------------
@app.route("/api/symptoms")
def api_symptoms():

    with get_db() as conn:

        rows = conn.execute('''
            SELECT symptom, severity, note, timestamp
            FROM symptoms
            ORDER BY timestamp DESC
        ''').fetchall()

    return jsonify([dict(r) for r in rows])

# ---------------------------------------------------
# ERRORS
# ---------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

# ---------------------------------------------------
# RUN APP
# ---------------------------------------------------
if __name__ == "__main__":

    init_db()

    app.run(debug=True)