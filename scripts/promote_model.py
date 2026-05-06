import os
import mlflow


def promote_model():

    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("Dagshub Token environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        "https://dagshub.com/unikbahadur1852/Laptop-Price-Project.mlflow"
    )

    client = mlflow.MlflowClient()
    model_name = "my_model"

    # -----------------------------
    # 1. Get current Production alias
    # -----------------------------
    try:
        prod_version = client.get_model_version_by_alias(model_name, "Production").version
    except:
        prod_version = None

    # -----------------------------
    # 2. Get Staging alias
    # -----------------------------
    try:
        staging_version = client.get_model_version_by_alias(model_name, "Staging").version
    except:
        raise ValueError("No Staging model found")

    # -----------------------------
    # 3. Move Production → Staging
    # -----------------------------
    if prod_version:
        client.set_registered_model_alias(
            name=model_name,
            alias="Staging",
            version=prod_version
        )

    # -----------------------------
    # 4. Promote Staging → Production
    # -----------------------------
    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=staging_version
    )

    print(f"Staging v{staging_version} → Production")
    print("Promotion complete 🚀")


if __name__ == "__main__":
    promote_model()