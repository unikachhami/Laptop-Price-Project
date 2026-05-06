import os
import mlflow


def promote_model():
    # -----------------------------
    # Auth setup
    # -----------------------------
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("Dagshub Token environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "unikbahadur1852"
    repo_name = "Laptop-Price-Project"

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    client = mlflow.MlflowClient()
    model_name = "my_model"

    # -----------------------------
    # 1. Get current Production version
    # -----------------------------
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])

    prod_version = prod_versions[0].version if prod_versions else None

    # -----------------------------
    # 2. Get latest Staging version safely
    # -----------------------------
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

    if not staging_versions:
        raise ValueError("No model found in Staging to promote.")

    staging_version = staging_versions[0].version

    # -----------------------------
    # 3. Move current Production → Staging (if exists)
    # -----------------------------
    if prod_version:
        client.transition_model_version_stage(
            name=model_name,
            version=prod_version,
            stage="Staging"
        )
        print(f"Production v{prod_version} → Staging")

    # -----------------------------
    # 4. Promote Staging → Production
    # -----------------------------
    client.transition_model_version_stage(
        name=model_name,
        version=staging_version,
        stage="Production"
    )

    print(f"Staging v{staging_version} → Production")
    print("Promotion complete ")


if __name__ == "__main__":
    promote_model()