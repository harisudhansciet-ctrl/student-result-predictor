from flask import Flask, request
import joblib
import os

app = Flask(__name__)
model = joblib.load("model.pkl")
VERSION = os.getenv("MODEL_VERSION", "1.0")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        try:
            hours = float(request.form["hours"])
            attendance = float(request.form["attendance"])
            marks = float(request.form["marks"])
            if not (0 <= hours <= 10 and 0 <= attendance <= 100 and 0 <= marks <= 100):
                raise ValueError()
            prediction = model.predict([[hours, attendance, marks]])[0]
            result = "Pass" if prediction == 1 else "Fail"
        except (ValueError, TypeError, KeyError):
            result = "Enter valid values within the allowed ranges."

    return f"""
    <h2>Student Result Predictor</h2>
    <p>Model Version: {VERSION}</p>
    <form method="POST">
      Study Hours (0-10): <input type="number" name="hours" min="0" max="10" step="any" required><br>
      Attendance (0-100): <input type="number" name="attendance" min="0" max="100" step="any" required><br>
      Internal Marks (0-100): <input type="number" name="marks" min="0" max="100" step="any" required><br>
      <button type="submit">Predict Result</button>
    </form>
    <h3>Prediction: {result}</h3>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
