import unittest
import mlflow
import os
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import r2_score, mean_squared_error


class TestLaptopPriceModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # ---------------- DAGSHUB AUTH ----------------
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "unikbahadur1852"
        repo_name = "Laptop-Price-Project"

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # ---------------- MODEL LOAD ----------------
        cls.model_name = "plmodel"
        client = mlflow.MlflowClient()

        versions = client.get_latest_versions(cls.model_name, stages=["Production"])

        if not versions:
            versions = client.get_latest_versions(cls.model_name, stages=["Staging"])

        if not versions:
            raise Exception("No model found in Production or Staging")

        cls.model_version = versions[0].version
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # ---------------- OPTIONAL: FEATURE PIPELINE ----------------
        # if you saved preprocessing objects
        try:
            cls.preprocessor = pickle.load(open("models/preprocessor.pkl", "rb"))
        except:
            cls.preprocessor = None

        # ---------------- HOLDOUT DATA ----------------
        cls.data = pd.read_csv("data/processed/test.csv")

    # ---------------- 1. MODEL LOADED ----------------
    def test_model_loaded(self):
        self.assertIsNotNone(self.model)

    # ---------------- 2. SIGNATURE TEST ----------------
    def test_model_prediction_signature(self):

        sample = self.data.iloc[[0]].copy()

        # remove target column if exists
        if "Price" in sample.columns:
            sample = sample.drop("Price", axis=1)

        pred = self.model.predict(sample)

        self.assertEqual(len(pred), 1)
        self.assertTrue(np.issubdtype(type(pred[0]), np.number))

    # ---------------- 3. PERFORMANCE TEST ----------------
    def test_model_performance(self):

        df = self.data.copy()

        # target column (change if yours is different)
        target_col = "Price"

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        preds = self.model.predict(X)

        r2 = r2_score(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))

        print("\nR2 Score:", r2)
        print("RMSE:", rmse)

        # ---------------- THRESHOLDS ----------------
        self.assertGreaterEqual(r2, 0.6, "R2 should be at least 0.6")
        self.assertLessEqual(rmse, 0.7 * np.std(y), "RMSE too high")


if __name__ == "__main__":
    unittest.main()