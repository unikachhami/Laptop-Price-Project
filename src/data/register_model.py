import mlflow
import json
import logging
from mlflow.tracking import MlflowClient
import dagshub


mlflow.set_tracking_uri(
    "https://dagshub.com/unikbahadur1852/Laptop-Price-Project.mlflow"
)
dagshub.init(
    repo_owner="unikbahadur1852",
    repo_name="Laptop-Price-Project",
    mlflow=True
)


def load_model(file_path: str):
    try:
        with open(file_path, "r") as file:
            model = json.load(file)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

import mlflow
from mlflow.tracking import MlflowClient


def register_model(model, model_name: str):
    try:
        # 1. Log model and register
        result = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="models",
            registered_model_name=model_name
        )

        client = MlflowClient()

        # 2. Get latest version (safe)
        latest_version = client.get_latest_versions(
            name=model_name,
            stages=["None"]
        )[0].version

        # 3. Promote
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production"
        )

        client.set_registered_model_tag(
            name=model_name,
            key="environment",
            value="production"
        )

        return latest_version

    except Exception as e:
        print(f"Registration failed: {e}")
        raise


    


def main():
    try:
        
        model = load_model("models/model.pkl")  

        model_name = "my_model"

        register_model(model, model_name)

    except Exception as e:
        logging.error(f"Failed pipeline: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()


        
