import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the saved model
MODEL_PATH = "linear_model.pickel"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# HTML Layout with Embedded CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .card {
            background: #ffffff;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
            width: 100%;
            max-width: 500px;
        }
        h2 {
            text-align: center;
            color: #333;
            margin-bottom: 24px;
            font-size: 26px;
        }
        .form-group {
            margin-bottom: 18px;
        }
        label {
            display: block;
            margin-bottom: 6px;
            color: #555;
            font-weight: 600;
            font-size: 14px;
        }
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e1e1;
            border-radius: 8px;
            font-size: 15px;
            transition: border-color 0.3s ease;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            width: 100%;
            background: #667eea;
            color: white;
            border: none;
            padding: 14px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 10px;
            transition: background 0.3s ease;
        }
        button:hover {
            background: #5a67d8;
        }
        .result-box {
            margin-top: 25px;
            padding: 15px;
            background: #eef2ff;
            border-left: 5px solid #667eea;
            border-radius: 6px;
            text-align: center;
        }
        .result-box h3 {
            color: #2b6cb0;
            font-size: 18px;
        }
        .result-box p {
            font-size: 24px;
            font-weight: bold;
            color: #4c51bf;
            margin-top: 5px;
        }
    </style>
</head>
<body>

<div class="card">
    <h2>Performance Predictor</h2>
    <form action="/predict" method="post">
        <div class="form-group">
            <label for="hours_studied">Hours Studied</label>
            <input type="number" step="any" name="hours_studied" id="hours_studied" required placeholder="e.g. 7">
        </div>
        <div class="form-group">
            <label for="previous_scores">Previous Scores</label>
            <input type="number" step="any" name="previous_scores" id="previous_scores" required placeholder="e.g. 85">
        </div>
        <div class="form-group">
            <label for="extracurricular">Extracurricular Activities</label>
            <select name="extracurricular" id="extracurricular" required>
                <option value="1">Yes</option>
                <option value="0">No</option>
            </select>
        </div>
        <div class="form-group">
            <label for="sleep_hours">Sleep Hours</label>
            <input type="number" step="any" name="sleep_hours" id="sleep_hours" required placeholder="e.g. 8">
        </div>
        <div class="form-group">
            <label for="sample_papers">Sample Question Papers Practiced</label>
            <input type="number" step="any" name="sample_papers" id="sample_papers" required placeholder="e.g. 5">
        </div>
        <button type="submit">Predict Score</button>
    </form>

    {% if prediction is not none %}
    <div class="result-box">
        <h3>Predicted Index / Score</h3>
        <p>{{ prediction }}</p>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract inputs from form
        hours_studied = float(request.form["hours_studied"])
        previous_scores = float(request.form["previous_scores"])
        extracurricular = float(request.form["extracurricular"])
        sleep_hours = float(request.form["sleep_hours"])
        sample_papers = float(request.form["sample_papers"])

        # Arrange features into a 2D array matching training feature order
        features = np.array([[hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers]])

        # Perform prediction
        raw_prediction = model.predict(features)[0]
        formatted_prediction = round(float(raw_prediction), 2)

        return render_template_string(HTML_TEMPLATE, prediction=formatted_prediction)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
