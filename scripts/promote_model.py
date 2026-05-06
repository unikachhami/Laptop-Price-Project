import os
import mlflow
from mlflow.tracking import MlflowClient


def promote_model():

    token = os.getenv("DAGSHUB_TOKEN")
    if not token:
        raise EnvironmentError("DAGSHUB_TOKEN not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(
        "https://dagshub.com/unikbahadur1852/Laptop-Price-Project.mlflow"
    )

    client = MlflowClient()
    model_name = "plmodel"

    # -----------------------------
    # 1. get latest version (NEW MODEL)
    # -----------------------------
    latest_versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    new_model_version = max(
        latest_versions,
        key=lambda x: int(x.version)
    ).version

    # -----------------------------
    # 2. get current production
    # -----------------------------
    try:
        prod = client.get_model_version_by_alias(
            model_name, "Production"
        ).version
    except:
        prod = None

    # -----------------------------
    # 3. move current prod → staging
    # -----------------------------
    if prod:
        client.set_registered_model_alias(
            name=model_name,
            alias="Staging",
            version=prod
        )

    # -----------------------------
    # 4. promote NEW → production
    # -----------------------------
    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=new_model_version
    )

    print(f" Version {new_model_version} promoted to Production")

if __name__ == "__main__":
    promote_model()