from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import mlflow
import os

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)

# -----------------------------
# MLflow setup
# -----------------------------
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("Environment variable is not set:")

os.environ["MLFLOW_TRACKING_USERNAME"]= dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = 'https:/dagshub.com'
repo_owner = 'unikbahadur1852'
repo_name = 'Laptop-Price-Project'


mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}')

MODEL_NAME = "lpmodel"


# -----------------------------
# Load latest production model
# -----------------------------
def load_model():
    client = mlflow.MlflowClient()

    # Try production first
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

    if not versions:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])

    if not versions:
        versions = client.get_latest_versions(MODEL_NAME, stages=["None"])

    version = versions[0].version

    model_uri = f"models:/{MODEL_NAME}/{version}"
    print("🚀 Loading model:", model_uri)

    return mlflow.pyfunc.load_model(model_uri)


model = load_model()


# -----------------------------
# HOME
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html", result=None)


# -----------------------------
# PREDICT
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # ---------------- INPUT ----------------
        company = request.form["company"]
        typename = request.form["type"]
        cpu = request.form["cpu"]
        gpu = request.form["gpu"]
        os_name = request.form["os"]

        ram = int(request.form["ram"])
        weight = float(request.form["weight"])
        screen_size = float(request.form["screen_size"])

        resolution = request.form["resolution"]
        x_res, y_res = resolution.split("x")
        x_res = int(x_res)
        y_res = int(y_res)

        hdd = int(request.form["hdd"])
        ssd = int(request.form["ssd"])

        touchscreen = 1 if request.form["touchscreen"] == "Yes" else 0
        ips = 1 if request.form["ips"] == "Yes" else 0

        # ---------------- FEATURE ENGINEERING ----------------
        ppi = ((x_res**2 + y_res**2) ** 0.5) / screen_size

        # ---------------- BUILD INPUT DATAFRAME ----------------
        input_df = pd.DataFrame([{
            "Company": company,
            "TypeName": typename,
            "Cpu brand": cpu,
            "Gpu Brand": gpu,
            "os": os_name,
            "Ram": ram,
            "Weight": weight,
            "Touchscreen": touchscreen,
            "IPS": ips,
            "PPI": ppi,
            "HDD": hdd,
            "SSD": ssd
        }])

        # ---------------- PREDICTION (PIPELINE MAGIC) ----------------
        pred = model.predict(input_df)[0]

        # reverse log transform
        pred = np.exp(pred)

        return render_template("index.html", result=int(pred))

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)