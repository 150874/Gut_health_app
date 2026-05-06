from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# -----------------------------
# DATABASE SETUP
# -----------------------------
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS medications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            time TEXT
        )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS symptoms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symptom TEXT,
        note TEXT,
        time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

    conn.commit()
    conn.close()


# -----------------------------
# SIMPLE FOOD RULES DATABASE
# -----------------------------
foods = {
    "coffee": {"gastritis": "avoid", "gerd": "avoid"},
    "banana": {"gastritis": "safe", "gerd": "safe"},
    "spicy food": {"gastritis": "avoid", "gerd": "avoid"},
    "tea": {"gastritis": "safe", "gerd": "neutral"}
}


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route('/')
def home():
    return render_template("index.html")


# -----------------------------
# FOOD CHECK LOGIC
# -----------------------------
@app.route('/check', methods=['POST'])
def check():
    food = request.form['food'].lower()
    condition = request.form['condition']

    result = foods.get(food, {}).get(condition, "unknown")

    return f"""
        <h3>{food} is {result} for {condition}</h3>
        <a href="/">Go back</a>
    """


# -----------------------------
# MEDICATION PAGE
# -----------------------------
@app.route('/medication')
def medication():
    return '''
        <h2>Add Medication</h2>
        <form action="/add_med" method="post">
            Name: <input type="text" name="name"><br><br>
            Time: <input type="time" name="time"><br><br>
            <button type="submit">Add</button>
        </form>
        <br>
        <a href="/view_meds">View Medications</a>
        <br><a href="/">Home</a>
    '''


# -----------------------------
# ADD MEDICATION (SAVE TO DB)
# -----------------------------
@app.route('/add_med', methods=['POST'])
def add_med():
    name = request.form['name']
    time = request.form['time']

    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute(
        "INSERT INTO medications (name, time) VALUES (?, ?)",
        (name, time)
    )

    conn.commit()
    conn.close()

    return "<h3>Medication saved!</h3><a href='/medication'>Go back</a>"


# -----------------------------
# VIEW MEDICATIONS
# -----------------------------
@app.route('/view_meds')
def view_meds():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute("SELECT * FROM medications")
    meds = c.fetchall()

    conn.close()

    output = "<h2>Your Medications</h2>"

    for med in meds:
        output += f"<p>{med[1]} at {med[2]}</p>"

    output += "<br><a href='/medication'>Add more</a>"
    return output


# -----------------------------
# SYMPTOMS
# -----------------------------

@app.route('/symptoms')
def symptoms():
    return '''
        <h2>Log Symptoms</h2>
        <form action="/add_symptom" method="post">
            Symptom:
            <input type="text" name="symptom"><br><br>

            Note:
            <input type="text" name="note"><br><br>

            <button type="submit">Save</button>
        </form>

        <br>
        <a href="/view_symptoms">View Symptoms</a>
        <br>
        <a href="/">Home</a>
    '''

@app.route('/add_symptom', methods=['POST'])
def add_symptom():
    symptom = request.form['symptom']
    note = request.form['note']

    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute(
        "INSERT INTO symptoms (symptom, note) VALUES (?, ?)",
        (symptom, note)
    )

    conn.commit()
    conn.close()

    return "<h3>Symptom saved!</h3><a href='/symptoms'>Go back</a>"

@app.route('/view_symptoms')
def view_symptoms():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute("SELECT * FROM symptoms ORDER BY id DESC")
    data = c.fetchall()

    conn.close()

    output = "<h2>Your Symptoms</h2>"

    for row in data:
        output += f"<p><b>{row[1]}</b> - {row[2]} <small>({row[3]})</small></p>"

    output += "<br><a href='/symptoms'>Add more</a>"
    return output

#----------------------------
#ANALYSIS
#-----------------------------

@app.route('/insights')
def insights():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Get symptoms
    c.execute("SELECT symptom FROM symptoms")
    symptoms = c.fetchall()

    conn.close()

    count_map = {}

    for s in symptoms:
        symptom = s[0].lower()
        if symptom in count_map:
            count_map[symptom] += 1
        else:
            count_map[symptom] = 1

    output = "<h2>Health Insights</h2>"

    if not count_map:
        output += "<p>No data yet. Start logging symptoms.</p>"
    else:
        for key, value in count_map.items():
            if value >= 3:
                output += f"<p>⚠️ {key} appears frequently ({value} times)</p>"
            else:
                output += f"<p>{key}: {value} times</p>"

    output += "<br><a href='/'>Home</a>"
    return output

reminders = []

@app.route('/set_reminder', methods=['POST'])
def set_reminder():
    name = request.form['name']
    time = request.form['time']

    reminders.append({"name": name, "time": time})

    return "<h3>Reminder set (demo)</h3><a href='/reminders'>Back</a>"

# -----------------------------
# START APP
# -----------------------------
if __name__ == '__main__':
    init_db()
    app.run(debug=True)