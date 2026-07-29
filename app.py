import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the AdaBoost model
MODEL_PATH = "adaboost.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Feature metadata extracted from your model
FEATURE_CONFIG = [
    {
        "name": "Age",
        "label": "Age",
        "type": "number",
        "default": 35,
        "step": 1,
        "min": 18,
        "max": 100,
        "desc": "Customer's age in years",
    },
    {
        "name": "Gender",
        "label": "Gender",
        "type": "select",
        "options": [("0", "Female"), ("1", "Male")],
        "default": "0",
        "desc": "Gender identity",
    },
    {
        "name": "Tenure",
        "label": "Tenure (Months)",
        "type": "number",
        "default": 12,
        "step": 1,
        "min": 0,
        "max": 120,
        "desc": "Months as a customer",
    },
    {
        "name": "Usage Frequency",
        "label": "Usage Frequency",
        "type": "number",
        "default": 15,
        "step": 1,
        "min": 0,
        "max": 30,
        "desc": "Monthly usage frequency count",
    },
    {
        "name": "Support Calls",
        "label": "Support Calls",
        "type": "number",
        "default": 2,
        "step": 1,
        "min": 0,
        "max": 20,
        "desc": "Number of customer support calls",
    },
    {
        "name": "Payment Delay",
        "label": "Payment Delay (Days)",
        "type": "number",
        "default": 5,
        "step": 1,
        "min": 0,
        "max": 60,
        "desc": "Delay in payment in days",
    },
    {
        "name": "Subscription Type",
        "label": "Subscription Type",
        "type": "select",
        "options": [("0", "Basic"), ("1", "Standard"), ("2", "Premium")],
        "default": "1",
        "desc": "Subscription tier level",
    },
    {
        "name": "Contract Length",
        "label": "Contract Length",
        "type": "select",
        "options": [("0", "Monthly"), ("1", "Quarterly"), ("2", "Annual")],
        "default": "0",
        "desc": "Contract duration period",
    },
    {
        "name": "Total Spend",
        "label": "Total Spend ($)",
        "type": "number",
        "default": 500,
        "step": 10,
        "min": 0,
        "max": 10000,
        "desc": "Cumulative monetary spend",
    },
    {
        "name": "Last Interaction",
        "label": "Last Interaction (Days ago)",
        "type": "number",
        "default": 10,
        "step": 1,
        "min": 0,
        "max": 30,
        "desc": "Days since last platform interaction",
    },
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost AI Predictor</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Particles.js -->
    <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>

    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.12);
            --accent-purple: #a855f7;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --card-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: #0f172a;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            position: relative;
            overflow-x: hidden;
        }

        #particles-js {
            position: fixed;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: 1;
            pointer-events: none;
        }

        .container {
            width: 100%;
            max-width: 1200px;
            z-index: 2;
            position: relative;
        }

        /* Glassmorphism Header */
        header {
            text-align: center;
            margin-bottom: 2rem;
            animation: fadeInDown 0.8s ease-out;
        }

        header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #a855f7, #6366f1, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 1rem;
        }

        /* Main Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 900px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Glass Card */
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: var(--card-shadow);
            animation: fadeInUp 0.8s ease-out;
            transition: transform 0.3s ease, border-color 0.3s ease;
        }

        .glass-card:hover {
            border-color: rgba(168, 85, 247, 0.3);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            font-size: 1.25rem;
            font-weight: 600;
            color: #fff;
            border-bottom: 1px solid var(--glass-border);
            padding-bottom: 0.75rem;
        }

        .card-header i {
            color: var(--accent-cyan);
        }

        /* Form Inputs */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.2rem;
        }

        @media (max-width: 500px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        .input-wrapper {
            position: relative;
        }

        .input-control {
            width: 100%;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-control:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 15px rgba(168, 85, 247, 0.3);
            background: rgba(15, 23, 42, 0.8);
        }

        select.input-control {
            cursor: pointer;
        }

        select.input-control option {
            background: #1e1b4b;
            color: #fff;
        }

        /* Submit Button */
        .submit-btn {
            grid-column: span 2;
            margin-top: 1rem;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
            border: none;
            color: white;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 10px 25px rgba(168, 85, 247, 0.3);
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(168, 85, 247, 0.5);
            filter: brightness(1.1);
        }

        .submit-btn:active {
            transform: translateY(0);
        }

        /* Prediction Output Section */
        .results-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            height: 100%;
            justify-content: space-between;
        }

        .placeholder-text {
            text-align: center;
            color: var(--text-muted);
            margin: auto 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }

        .placeholder-text i {
            font-size: 3rem;
            color: var(--glass-border);
        }

        .result-box {
            background: rgba(15, 23, 42, 0.6);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid var(--glass-border);
            animation: pulseGlow 2s infinite alternate;
        }

        .result-value {
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.5rem;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .chart-container {
            position: relative;
            height: 260px;
            width: 100%;
        }

        /* Animations */
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulseGlow {
            from { box-shadow: 0 0 10px rgba(99, 102, 241, 0.1); }
            to { box-shadow: 0 0 25px rgba(168, 85, 247, 0.3); }
        }

        /* Spinner */
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <div id="particles-js"></div>

    <div class="container">
        <header>
            <h1>AdaBoost Intelligence Hub</h1>
            <p>Interactive Machine Learning Prediction Engine</p>
        </header>

        <div class="dashboard-grid">
            <!-- Form Card -->
            <div class="glass-card">
                <div class="card-header">
                    <i class="fa-solid fa-sliders"></i> Input Parameters
                </div>
                <form id="prediction-form" class="form-grid">
                    {% for feature in features %}
                    <div class="input-group">
                        <label for="{{ feature.name }}">{{ feature.label }}</label>
                        <div class="input-wrapper">
                            {% if feature.type == 'select' %}
                            <select class="input-control" id="{{ feature.name }}" name="{{ feature.name }}">
                                {% for val, text in feature.options %}
                                <option value="{{ val }}" {% if val == feature.default %}selected{% endif %}>{{ text }}</option>
                                {% endfor %}
                            </select>
                            {% else %}
                            <input type="number" 
                                   class="input-control" 
                                   id="{{ feature.name }}" 
                                   name="{{ feature.name }}" 
                                   value="{{ feature.default }}"
                                   step="{{ feature.step }}"
                                   min="{{ feature.min }}"
                                   max="{{ feature.max }}"
                                   required>
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}

                    <button type="submit" class="submit-btn" id="submit-btn">
                        <span id="btn-text">Generate Prediction</span>
                        <div class="spinner" id="btn-spinner"></div>
                        <i class="fa-solid fa-wand-magic-sparkles" id="btn-icon"></i>
                    </button>
                </form>
            </div>

            <!-- Visualization Card -->
            <div class="glass-card">
                <div class="card-header">
                    <i class="fa-solid fa-chart-pie"></i> Real-Time Analysis
                </div>
                <div class="results-container">
                    <div id="placeholder" class="placeholder-text">
                        <i class="fa-solid fa-brain"></i>
                        <p>Fill out the parameters and click 'Generate Prediction' to analyze outcomes.</p>
                    </div>

                    <div id="results-content" style="display: none; flex-direction: column; gap: 1.5rem; width: 100%;">
                        <div class="result-box">
                            <div>Predicted Outcome</div>
                            <div class="result-value" id="prediction-output">Class 0</div>
                        </div>

                        <div class="chart-container">
                            <canvas id="probabilityChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Particles.js background
        particlesJS('particles-js', {
            particles: {
                number: { value: 50, density: { enable: true, value_area: 800 } },
                color: { value: '#a855f7' },
                shape: { type: 'circle' },
                opacity: { value: 0.3, random: true },
                size: { value: 3, random: true },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#6366f1',
                    opacity: 0.2,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 1.5,
                    direction: 'none',
                    random: true,
                    straight: false,
                    out_mode: 'out',
                    bounce: false
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: { onhover: { enable: true, mode: 'grab' }, onclick: { enable: true, mode: 'push' } },
                modes: { grab: { distance: 140, line_linked: { opacity: 0.5 } } }
            }
        });

        // Global Chart Instance
        let probabilityChart = null;

        document.getElementById('prediction-form').addEventListener('submit', async function (e) {
            e.preventDefault();

            const submitBtn = document.getElementById('submit-btn');
            const btnText = document.getElementById('btn-text');
            const btnSpinner = document.getElementById('btn-spinner');
            const btnIcon = document.getElementById('btn-icon');

            // Show Loading State
            btnText.textContent = "Analyzing...";
            btnSpinner.style.display = "inline-block";
            btnIcon.style.display = "none";
            submitBtn.disabled = true;

            const formData = new FormData(this);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.error) {
                    alert(data.error);
                    return;
                }

                // Display Results
                document.getElementById('placeholder').style.display = 'none';
                const resultsContent = document.getElementById('results-content');
                resultsContent.style.display = 'flex';

                document.getElementById('prediction-output').textContent = `Class ${data.prediction}`;

                // Update or Create Chart
                renderChart(data.classes, data.probabilities);

            } catch (err) {
                console.error("Error during prediction request:", err);
                alert("Failed to fetch prediction from backend.");
            } finally {
                // Reset Button State
                btnText.textContent = "Generate Prediction";
                btnSpinner.style.display = "none";
                btnIcon.style.display = "inline-block";
                submitBtn.disabled = false;
            }
        });

        function renderChart(labels, probabilities) {
            const ctx = document.getElementById('probabilityChart').getContext('2d');

            if (probabilityChart) {
                probabilityChart.destroy();
            }

            probabilityChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels.map(l => `Class ${l}`),
                    datasets: [{
                        label: 'Probability Confidence',
                        data: probabilities,
                        backgroundColor: [
                            'rgba(168, 85, 247, 0.7)',
                            'rgba(6, 182, 212, 0.7)',
                            'rgba(99, 102, 241, 0.7)'
                        ],
                        borderColor: [
                            '#a855f7',
                            '#06b6d4',
                            '#6366f1'
                        ],
                        borderWidth: 2,
                        borderRadius: 12
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            ticks: { color: '#94a3b8' },
                            grid: { color: 'rgba(255, 255, 255, 0.08)' }
                        },
                        x: {
                            ticks: { color: '#94a3b8' },
                            grid: { display: false }
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE, features=FEATURE_CONFIG)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file not found or corrupted."}), 500

    try:
        # Extract features in exact order expected by the model
        input_data = []
        for feature in FEATURE_CONFIG:
            val = request.form.get(feature["name"])
            input_data.append(float(val))

        features_array = np.array([input_data])
        
        # Make prediction
        prediction = int(model.predict(features_array)[0])
        
        # Calculate probabilities
        probabilities = model.predict_proba(features_array)[0].tolist()
        classes = [int(c) for c in model.classes_]

        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities,
            "classes": classes
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
