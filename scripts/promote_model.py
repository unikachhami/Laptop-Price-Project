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
    # Get current production model
    # -----------------------------
    try:
        prod_model = client.get_model_version_by_alias(
            model_name,
            "Production"
        )

        # move old production → staging
        client.set_registered_model_alias(
            name=model_name,
            alias="Staging",
            version=prod_model.version
        )

    except Exception:
        # no production model yet
        pass

    # -----------------------------
    # Get latest staging model
    # -----------------------------
    staging_model = client.get_model_version_by_alias(
        model_name,
        "Staging"
    )

    # promote staging → production
    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=staging_model.version
    )

    print(f"Promoted version {staging_model.version} → Production")


if __name__ == "__main__":
    promote_model()