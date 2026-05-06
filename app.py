from flask import Flask, render_template, request

app = Flask(__name__)

foods = {
    "coffee": {"gastritis": "avoid", "gerd": "avoid"},
    "banana": {"gastritis": "safe", "gerd": "safe"},
    "spicy food": {"gastritis": "avoid", "gerd": "avoid"}
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/check', methods=['POST'])
def check():
    food = request.form['food'].lower()
    condition = request.form['condition']

    result = foods.get(food, {}).get(condition, "unknown")

    return f"{food} is {result} for {condition}"

if __name__ == '__main__':
    app.run(debug=True)

    medications = []

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
    '''
@app.route('/add_med', methods=['POST'])
def add_med():
    name = request.form['name']
    time = request.form['time']

    medications.append({"name": name, "time": time})

    return "<h3>Medication added!</h3><a href='/medication'>Go back</a>"

@app.route('/view_meds')
def view_meds():
    output = "<h2>Your Medications</h2>"

    for med in medications:
        output += f"<p>{med['name']} at {med['time']}</p>"

    output += "<br><a href='/medication'>Add more</a>"
    return output