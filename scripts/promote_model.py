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
    # 1. get current production version
    # -----------------------------
    try:
        prod_version = client.get_model_version_by_alias(
            model_name, "Production"
        ).version
    except:
        prod_version = None

    # -----------------------------
    # 2. get all versions
    # -----------------------------
    all_versions = client.search_model_versions(f"name='{model_name}'")

    # -----------------------------
    # 3. pick latest NON-production version
    # -----------------------------
    candidates = [
        v for v in all_versions
        if v.version != prod_version
    ]

    if not candidates:
        raise ValueError("No new model version available to promote.")

    new_model_version = max(candidates, key=lambda x: int(x.version)).version

    # -----------------------------
    # 4. move current prod → staging
    # -----------------------------
    if prod_version:
        client.set_registered_model_alias(
            name=model_name,
            alias="Staging",
            version=prod_version
        )

    # -----------------------------
    # 5. promote new → production
    # -----------------------------
    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=new_model_version
    )

    print(f" Version {prod_version} → Staging")
    print(f" Version {new_model_version} → Production")


if __name__ == "__main__":
    promote_model()