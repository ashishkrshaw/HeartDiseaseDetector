from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Favicon handler to suppress 404 errors
@app.route('/favicon.ico')
def favicon():
    return '', 204


def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@app.route("/")
def index():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return render_template("index.html", result="Model not found. Please run /train first.")

    try:
        # expected feature order: age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose
        features = [
            float(request.form.get("age", 0)),
            float(request.form.get("cigsPerDay", 0)),
            float(request.form.get("totChol", 0)),
            float(request.form.get("sysBP", 0)),
            float(request.form.get("diaBP", 0)),
            float(request.form.get("BMI", 0)),
            float(request.form.get("heartRate", 0)),
            float(request.form.get("glucose", 0))
        ]
    except ValueError:
        return render_template("index.html", result="Invalid input. Please enter numeric values.")

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0, 1]
    pred = int(proba >= 0.5)

    result = {
        "probability": float(proba),
        "prediction": int(pred)
    }
    return render_template("index.html", result=result)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json() or {}
        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            return jsonify({'error': 'Model files not found in app/models/ folder. Please ensure model files are copied.'}), 500

        # Features in correct order: age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose
        features = [
            float(data.get("age", 0)),
            float(data.get("cigsPerDay", 0)),
            float(data.get("totChol", 0)),
            float(data.get("sysBP", 0)),
            float(data.get("diaBP", 0)),
            float(data.get("BMI", 0)),
            float(data.get("heartRate", 0)),
            float(data.get("glucose", 0))
        ]
        
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)[0, 1]
        pred = int(proba >= 0.5)
        return jsonify({'probability': float(proba), 'prediction': pred})
    
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid input values. Please enter numeric values. Details: {str(e)}'}), 400
    except Exception as e:
        print(f"API Error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route("/train", methods=["GET"])
def train():
    # Run the training script to create model artifacts
    train_script = os.path.join(BASE_DIR, "train_model.py")
    if not os.path.exists(train_script):
        return "Train script not found."

    # import and run
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_model", train_script)
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)

    # tm.train_and_save() should exist
    if hasattr(tm, "train_and_save"):
        msg = tm.train_and_save()
        return redirect(url_for("index"))
    else:
        return "train_and_save function not found in train_model.py"


@app.route('/api/train', methods=['POST'])
def api_train():
    train_script = os.path.join(BASE_DIR, "train_model.py")
    if not os.path.exists(train_script):
        return jsonify({'error': 'Train script not found.'}), 500

    import importlib.util
    spec = importlib.util.spec_from_file_location("train_model", train_script)
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)

    if hasattr(tm, "train_and_save"):
        try:
            tm.train_and_save()
            return jsonify({'status': 'ok', 'message': 'Training completed and model saved.'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'train_and_save function not found in train_model.py'}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
