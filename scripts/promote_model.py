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
    # MLflow tracking URI
    # -----------------------------
    dagshub_url = "https://dagshub.com"
    repo_owner = "unikbahadur1852"
    repo_name = "Laptop-Price-Project"

    mlflow.set_tracking_uri(
        f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
    )

    client = MlflowClient()

    model_name = "plmodel"

    # -----------------------------
    # Get staging model via alias
    # -----------------------------
    staging_version = client.get_model_version_by_alias(
        model_name,
        "Staging"
    ).version

    # -----------------------------
    # Archive current production models
    # -----------------------------
    try:
        prod_model = client.get_model_version_by_alias(
            model_name,
            "Production"
        )

        client.set_registered_model_alias(
            name=model_name,
            alias="Archived",
            version=prod_model.version
        )

    except Exception:
        # no production model yet (first run)
        pass

    # -----------------------------
    # Promote staging → production
    # -----------------------------
    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=staging_version
    )

    print(f" Model version {staging_version} promoted to Production")


if __name__ == "__main__":
    promote_model()