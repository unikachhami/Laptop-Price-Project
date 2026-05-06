import os
import mlflow


def promote_model():

    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("Dagshub Token not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        "https://dagshub.com/unikbahadur1852/Laptop-Price-Project.mlflow"
    )

    client = mlflow.MlflowClient()
    model_name = "plmodel"   

    try:
        # 1. Get the current Staging version
        staging_version_info = client.get_model_version_by_alias(model_name, "Staging")
        v_num = staging_version_info.version
    except Exception:
        print("Error: No model found with 'Staging' alias. Run model_building.py first.")
        return

    # 2. Promote this version to Production
    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=v_num
    )

    # 3. FIX: Remove the Staging alias from this version 
    # This prevents Version 24 from having TWO aliases at once.
    client.delete_registered_model_alias(model_name, "Staging")

    print(f"Successfully promoted Version {v_num} to Production and cleared Staging.")

if __name__ == "__main__":
    promote_model()