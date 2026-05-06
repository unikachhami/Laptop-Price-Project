import os
import dagshub
import mlflow
def promotr_model():
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("Environment variable is not set:")

    os.environ["MLFLOW_USERNAME"] = dagshub_token
    os.environ["MLFLOW_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "unikbahadur1852"
    repo_name = "Laptop-Price-Project"

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}")

    client = mlflow.MlflowClient()

    my_model = 'pl_model'

    latest_version_staging = client.get_latest_versions(my_model,stages=['Staging'])[0]._version

    prod_verion = client.get_latest_versions(my_model,stages=['Production'])

    for version in prod_verion:
        client.set_registered_model_alias(name=my_model,version=version.version,alias=['Archived'])

    client.set_registered_model_alias(my_model,version=latest_version_staging,alias=['Production'])
    print(f"Model version {latest_version_staging} promoted to production")
