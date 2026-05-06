import os
import mlflow
from mlflow.tracking import MlflowClient


def promote_model():

    # -----------------------------
    # Auth
    # -----------------------------
    token = os.getenv("DAGSHUB_TOKEN")
    if not token:
        raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # -----------------------------
    # Tracking URI
    # -----------------------------
    mlflow.set_tracking_uri(
        "https://dagshub.com/unikbahadur1852/Laptop-Price-Project.mlflow"
    )

    client = MlflowClient()
    model_name = "plmodel"

    # -----------------------------
    # Get staging version safely
    # -----------------------------
    try:
        staging_version = client.get_model_version_by_alias(
            model_name,
            "Staging"
        ).version
    except Exception:
        raise Exception("No Staging model found. Cannot promote.")

    # -----------------------------
    # Archive current production (if exists)
    # -----------------------------
    try:
        prod_version = client.get_model_version_by_alias(
            model_name,
            "Production"
        ).version

        # move old production to archived
        client.set_registered_model_alias(
            name=model_name,
            alias="Archived",
            version=prod_version
        )

    except Exception:
        # no production yet
        pass

    # -----------------------------
    # IMPORTANT FIX:
    # Ensure Staging ≠ Production visually
    # (optional cleanup step)
    # -----------------------------
    client.set_registered_model_alias(
        name=model_name,
        alias="Staging",
        version=staging_version
    )

    # -----------------------------
    # Promote to Production
    # -----------------------------
    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=staging_version
    )

    print(f" Version {staging_version} promoted to Production")


if __name__ == "__main__":
    promote_model()