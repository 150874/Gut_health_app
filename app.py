from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import sqlite3
from datetime import datetime
from contextlib import contextmanager
import logging

# -----------------------------
# APP SETUP
# -----------------------------
app = Flask(__name__)
app.secret_key = "change-this-in-production"  # Required for flash messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "database.db"

# Valid condition columns — used to prevent SQL injection via f-string
CONDITION_COLUMN_MAP = {
    "gastritis": "gastritis_status",
    "gerd":      "gerd_status",
    "hpylori":   "hpylori_status",
}

# -----------------------------
# DATABASE LAYER
# -----------------------------

@contextmanager
def get_db():
    """Context manager for safe DB access with automatic cleanup."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # Enforce FK constraints
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize all tables and seed default data."""
    with get_db() as conn:
        c = conn.cursor()

        # 1. User Profile — stores active condition and display name
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_profile (
                id        INTEGER PRIMARY KEY CHECK (id = 1),  -- Singleton row
                name      TEXT    DEFAULT 'User',
                condition TEXT    DEFAULT 'gastritis'
                    CHECK (condition IN ('gastritis', 'gerd', 'hpylori'))
            )
        ''')

        # Insert default profile if missing
        c.execute("INSERT OR IGNORE INTO user_profile (id) VALUES (1)")

        # 2. Medications
        c.execute('''
            CREATE TABLE IF NOT EXISTS medications (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT    NOT NULL,
                dosage       TEXT,
                time         TEXT    NOT NULL,
                instructions TEXT
            )
        ''')

        # 3. Symptoms
        c.execute('''
            CREATE TABLE IF NOT EXISTS symptoms (
                id        INTEGER  PRIMARY KEY AUTOINCREMENT,
                symptom   TEXT     NOT NULL,
                severity  INTEGER  NOT NULL DEFAULT 5
                    CHECK (severity BETWEEN 1 AND 10),
                note      TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 4. Master Food Rules
        c.execute('''
            CREATE TABLE IF NOT EXISTS food_rules (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                food_name        TEXT    UNIQUE NOT NULL,
                gastritis_status TEXT    CHECK (gastritis_status IN ('safe', 'neutral', 'avoid')),
                gerd_status      TEXT    CHECK (gerd_status      IN ('safe', 'neutral', 'avoid')),
                hpylori_status   TEXT    CHECK (hpylori_status   IN ('safe', 'neutral', 'avoid'))
            )
        ''')

        # 5. Food Consumption Logs — FK to food_rules for referential integrity
        c.execute('''
            CREATE TABLE IF NOT EXISTS food_logs (
                id        INTEGER  PRIMARY KEY AUTOINCREMENT,
                food_name TEXT     NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (food_name) REFERENCES food_rules (food_name)
                    ON UPDATE CASCADE ON DELETE SET NULL
            )
        ''')

        # Seed food rules if empty
        c.execute("SELECT COUNT(*) FROM food_rules")
        if c.fetchone()[0] == 0:
            sample_foods = [
                ("Coffee",         "avoid",   "avoid",   "neutral"),
                ("Banana",         "safe",    "safe",    "safe"),
                ("Ginger Tea",     "safe",    "neutral", "safe"),
                ("Spicy Peppers",  "avoid",   "avoid",   "avoid"),
                ("Oatmeal",        "safe",    "safe",    "safe"),
                ("Yogurt",         "neutral", "safe",    "safe"),
                ("Broccoli",       "safe",    "safe",    "safe"),
                ("Alcohol",        "avoid",   "avoid",   "avoid"),
                ("Whole Milk",     "neutral", "avoid",   "neutral"),
                ("Green Tea",      "neutral", "neutral", "safe"),
                ("White Rice",     "safe",    "safe",    "safe"),
                ("Tomatoes",       "avoid",   "avoid",   "neutral"),
                ("Almonds",        "safe",    "neutral", "safe"),
                ("Fried Chicken",  "avoid",   "avoid",   "neutral"),
                ("Manuka Honey",   "safe",    "safe",    "safe"),
            ]
            c.executemany(
                "INSERT INTO food_rules "
                "(food_name, gastritis_status, gerd_status, hpylori_status) "
                "VALUES (?, ?, ?, ?)",
                sample_foods
            )

        logger.info("Database initialized successfully.")


# -----------------------------
# HELPERS
# -----------------------------

def get_user_profile():
    """Return the singleton user profile row."""
    with get_db() as conn:
        return conn.execute("SELECT * FROM user_profile WHERE id = 1").fetchone()


def validate_severity(raw_value: str, default: int = 5) -> int:
    """Parse and clamp severity to 1–10."""
    try:
        value = int(raw_value)
        return max(1, min(10, value))
    except (ValueError, TypeError):
        return default


# -----------------------------
# ROUTES: CORE
# -----------------------------

@app.route("/")
def home():
    profile = get_user_profile()
    return render_template("index.html", profile=profile)


@app.route("/profile", methods=["POST"])
def update_profile():
    """Update user name and active condition."""
    name      = request.form.get("name", "").strip() or "User"
    condition = request.form.get("condition", "gastritis")

    if condition not in CONDITION_COLUMN_MAP:
        flash("Invalid condition selected.", "error")
        return redirect(url_for("home"))

    with get_db() as conn:
        conn.execute(
            "UPDATE user_profile SET name = ?, condition = ? WHERE id = 1",
            (name, condition)
        )

    flash("Profile updated successfully.", "success")
    return redirect(url_for("home"))


# -----------------------------
# ROUTES: FOOD
# -----------------------------

@app.route("/check", methods=["POST"])
def check_food():
    """Check a food item against the user's active condition."""
    food_query = request.form.get("food", "").strip().capitalize()
    if not food_query:
        flash("Please enter a food name.", "error")
        return redirect(url_for("home"))

    # Honour explicit condition in form, or fall back to user profile
    condition = request.form.get("condition", "").lower()
    if condition not in CONDITION_COLUMN_MAP:
        profile   = get_user_profile()
        condition = profile["condition"]

    target_col = CONDITION_COLUMN_MAP[condition]   # Safe — validated above

    with get_db() as conn:
        row = conn.execute(
            f"SELECT food_name, {target_col} AS status FROM food_rules "
            "WHERE LOWER(food_name) = LOWER(?)",
            (food_query,)
        ).fetchone()

    status = row["status"] if row else "unknown — consult your dietitian"
    return render_template(
        "index.html",
        profile=get_user_profile(),
        food=food_query,
        status=status,
        condition=condition,
    )


@app.route("/food_rules")
def food_rules():
    """List all food rules, optionally filtered by condition status."""
    filter_status    = request.args.get("status")   # 'safe', 'neutral', 'avoid'
    filter_condition = request.args.get("condition", "gastritis")

    if filter_condition not in CONDITION_COLUMN_MAP:
        filter_condition = "gastritis"

    target_col = CONDITION_COLUMN_MAP[filter_condition]

    with get_db() as conn:
        if filter_status in ("safe", "neutral", "avoid"):
            rows = conn.execute(
                f"SELECT * FROM food_rules WHERE {target_col} = ? ORDER BY food_name",
                (filter_status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM food_rules ORDER BY food_name"
            ).fetchall()

    return render_template(
        "food_rules.html",
        rules=rows,
        condition=filter_condition,
        status_filter=filter_status,
    )


# -----------------------------
# ROUTES: MEAL LOGGING
# -----------------------------

@app.route("/log_meal", methods=["POST"])
def log_meal():
    food_name = request.form.get("food_name", "").strip().capitalize()
    if not food_name:
        flash("Food name cannot be empty.", "error")
        return redirect(url_for("home"))

    with get_db() as conn:
        # Verify food exists in rules before logging
        exists = conn.execute(
            "SELECT 1 FROM food_rules WHERE LOWER(food_name) = LOWER(?)",
            (food_name,)
        ).fetchone()

        if not exists:
            flash(
                f"'{food_name}' is not in the food database. "
                "Add it to food rules first.",
                "warning"
            )
            return redirect(url_for("home"))

        conn.execute("INSERT INTO food_logs (food_name) VALUES (?)", (food_name,))

    flash(f"Logged '{food_name}' successfully.", "success")
    return redirect(url_for("home"))


@app.route("/meal_history")
def meal_history():
    with get_db() as conn:
        logs = conn.execute(
            "SELECT fl.*, fr.gastritis_status, fr.gerd_status, fr.hpylori_status "
            "FROM food_logs fl "
            "LEFT JOIN food_rules fr ON LOWER(fl.food_name) = LOWER(fr.food_name) "
            "ORDER BY fl.timestamp DESC LIMIT 100"
        ).fetchall()
    return render_template("meal_history.html", logs=logs)


# -----------------------------
# ROUTES: SYMPTOMS
# -----------------------------

@app.route("/add_symptom", methods=["POST"])
def add_symptom():
    symptom  = request.form.get("symptom", "").strip()
    severity = validate_severity(request.form.get("severity", 5))
    note     = request.form.get("note", "").strip()

    if not symptom:
        flash("Symptom description cannot be empty.", "error")
        return redirect(url_for("view_symptoms"))

    with get_db() as conn:
        conn.execute(
            "INSERT INTO symptoms (symptom, severity, note) VALUES (?, ?, ?)",
            (symptom, severity, note)
        )

    flash("Symptom logged.", "success")
    return redirect(url_for("view_symptoms"))


@app.route("/view_symptoms")
def view_symptoms():
    with get_db() as conn:
        logs = conn.execute(
            "SELECT * FROM symptoms ORDER BY timestamp DESC"
        ).fetchall()
    return render_template("view_symptoms.html", logs=logs)


@app.route("/delete_symptom/<int:symptom_id>", methods=["POST"])
def delete_symptom(symptom_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM symptoms WHERE id = ?", (symptom_id,))
    flash("Symptom entry deleted.", "success")
    return redirect(url_for("view_symptoms"))


# -----------------------------
# ROUTES: MEDICATIONS
# -----------------------------

@app.route("/view_meds")
def view_meds():
    with get_db() as conn:
        meds = conn.execute(
            "SELECT * FROM medications ORDER BY time ASC"
        ).fetchall()
    return render_template("view_meds.html", meds=meds)


@app.route("/add_med", methods=["POST"])
def add_med():
    name         = request.form.get("name", "").strip()
    dosage       = request.form.get("dosage", "").strip()
    time_str     = request.form.get("time", "").strip()
    instructions = request.form.get("instructions", "").strip()

    if not name or not time_str:
        flash("Medication name and time are required.", "error")
        return redirect(url_for("view_meds"))

    # Validate time format (HH:MM)
    try:
        datetime.strptime(time_str, "%H:%M")
    except ValueError:
        flash("Time must be in HH:MM format.", "error")
        return redirect(url_for("view_meds"))

    with get_db() as conn:
        conn.execute(
            "INSERT INTO medications (name, dosage, time, instructions) "
            "VALUES (?, ?, ?, ?)",
            (name, dosage, time_str, instructions)
        )

    flash(f"Medication '{name}' added.", "success")
    return redirect(url_for("view_meds"))


@app.route("/delete_med/<int:med_id>", methods=["POST"])
def delete_med(med_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM medications WHERE id = ?", (med_id,))
    flash("Medication removed.", "success")
    return redirect(url_for("view_meds"))


# -----------------------------
# ROUTES: ANALYTICS & INSIGHTS
# -----------------------------

@app.route("/insights")
def insights():
    profile   = get_user_profile()
    condition = profile["condition"]
    target_col = CONDITION_COLUMN_MAP[condition]

    with get_db() as conn:
        # Correlation: "avoid" foods eaten within 4 hours before a symptom —
        # now filtered to the user's actual condition
        correlations = conn.execute(f'''
            SELECT
                s.symptom,
                s.severity,
                l.food_name,
                s.timestamp AS symptom_time,
                l.timestamp AS food_time
            FROM symptoms s
            JOIN food_logs l
                ON l.timestamp BETWEEN datetime(s.timestamp, "-4 hours")
                AND s.timestamp
            JOIN food_rules r
                ON LOWER(l.food_name) = LOWER(r.food_name)
            WHERE r.{target_col} = "avoid"
            ORDER BY s.timestamp DESC
            LIMIT 50
        ''').fetchall()

        # Most frequent symptoms (≥2 occurrences)
        frequency = conn.execute('''
            SELECT symptom, COUNT(*) AS count, ROUND(AVG(severity), 1) AS avg_severity
            FROM symptoms
            GROUP BY symptom
            HAVING count >= 2
            ORDER BY count DESC
        ''').fetchall()

        # Severity trend — last 14 days
        trend = conn.execute('''
            SELECT DATE(timestamp) AS day, ROUND(AVG(severity), 1) AS avg_severity
            FROM symptoms
            WHERE timestamp >= DATE("now", "-14 days")
            GROUP BY day
            ORDER BY day ASC
        ''').fetchall()

        # Most consumed foods
        top_foods = conn.execute('''
            SELECT food_name, COUNT(*) AS count
            FROM food_logs
            GROUP BY food_name
            ORDER BY count DESC
            LIMIT 10
        ''').fetchall()

    return render_template(
        "insights.html",
        correlations=correlations,
        frequency=frequency,
        trend=trend,
        top_foods=top_foods,
        condition=condition,
    )


# -----------------------------
# API ENDPOINTS (JSON)
# -----------------------------

@app.route("/api/food_check")
def api_food_check():
    """
    JSON endpoint for async food checks.
    GET /api/food_check?food=Banana&condition=gerd
    """
    food_query = request.args.get("food", "").strip().capitalize()
    condition  = request.args.get("condition", "gastritis").lower()

    if not food_query:
        return jsonify({"error": "Missing 'food' parameter"}), 400

    if condition not in CONDITION_COLUMN_MAP:
        return jsonify({"error": f"Invalid condition. Choose from: {list(CONDITION_COLUMN_MAP.keys())}"}), 400

    target_col = CONDITION_COLUMN_MAP[condition]

    with get_db() as conn:
        row = conn.execute(
            f"SELECT food_name, {target_col} AS status FROM food_rules "
            "WHERE LOWER(food_name) = LOWER(?)",
            (food_query,)
        ).fetchone()

    if row:
        return jsonify({"food": row["food_name"], "status": row["status"], "condition": condition})
    return jsonify({"food": food_query, "status": "unknown", "condition": condition})


@app.route("/api/symptoms")
def api_symptoms():
    """Return recent symptoms as JSON for charting."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT symptom, severity, note, timestamp FROM symptoms "
            "ORDER BY timestamp DESC LIMIT 50"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


# -----------------------------
# ERROR HANDLERS
# -----------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return render_template("500.html"), 500


# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    init_db()
    app.run(debug=True)