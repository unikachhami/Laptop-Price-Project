import unittest
import os
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np


class TestLaptopPriceModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # -----------------------------
        # Auth (CI-safe)
        # -----------------------------
        token = os.getenv("DAGSHUB_TOKEN")
        if not token:
            raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        mlflow.set_tracking_uri(
            "https://dagshub.com/unikbahadur1852/Laptop-Price-Project.mlflow"
        )

        cls.model_name = "plmodel"
        cls.client = MlflowClient()

        # -----------------------------
        # Load model via alias (NOT stages)
        # -----------------------------
        try:
            version = cls.client.get_model_version_by_alias(
                cls.model_name,
                "Staging"
            ).version

            model_uri = f"models:/{cls.model_name}/{version}"
            cls.model = mlflow.pyfunc.load_model(model_uri)

        except Exception as e:
            raise Exception(f"Model not found in Staging alias: {e}")

        # -----------------------------
        # Dummy test data (structure only)
        # -----------------------------
        cls.sample_input = pd.DataFrame([{
            "Company": "Apple",
            "TypeName": "Ultrabook",
            "Cpu brand": "Intel Core i5",
            "Gpu Brand": "Intel",
            "os": "Mac",
            "Ram": 8,
            "Weight": 1.5,
            "Touchscreen": 0,
            "IPS": 1,
            "PPI": 220,
            "HDD": 0,
            "SSD": 256
        }])

    # -----------------------------
    # TEST 1: model loads
    # -----------------------------
    def test_model_loaded(self):
        self.assertIsNotNone(self.model)

    # -----------------------------
    # TEST 2: prediction works
    # -----------------------------
    def test_prediction(self):
        pred = self.model.predict(self.sample_input)
        self.assertEqual(len(pred), 1)
        self.assertTrue(np.isfinite(pred[0]))


if __name__ == "__main__":
    unittest.main()