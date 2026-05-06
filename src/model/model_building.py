import json

# from fastapi.routing import _endpoint_context_cache
import dagshub.auth
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from src.logger import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from mlflow import MlflowClient
import pickle
import dagshub
import os

# mlflow.set_tracking_uri('https://dagshub.com/unikbahadur1852/Laptop-Price-Project.mlflow')
# dagshub.init(repo_owner='unikbahadur1852', repo_name='Laptop-Price-Project', mlflow=True)

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("Environment variable is not set:")

dagshub.auth.add_app_token(dagshub_token)

os.environ["MLFLOW_TRACKING_USERNAME"]= dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = 'https://dagshub.com'
repo_owner = 'unikbahadur1852'
repo_name = 'Laptop-Price-Project'



mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)



def load_data(data_path):
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Data Successfully loaded from {data_path}")
        return df
    except Exception as e:
        logging.error(f"Some error occurred {e}")
    



def build_pipeline(params, categorical_cols):
    try:

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )

        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            max_features=params['max_features'],
            max_samples=params['max_samples'],
            random_state=42
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        return pipeline
    except Exception as e:
        logging.error(f"Some error occurred {e}")



def save_pipeline(model, pipeline_path: str):
    try:
        with open(pipeline_path, 'wb') as file:
            pickle.dump(model, file)

        logging.info(f'Pipeline saved in {pipeline_path}')

    except Exception as e:
        logging.error(f'Some error occurred while saving pipeline: {e}')
        raise

def save_metrics(metrics,metrics_path)->dict:
    try:
        with open(metrics_path,'w')as path:
            json.dump(metrics,path,indent=2)
        logging.info(f'Model metrics saved in {metrics_path}')
    except Exception as e:
        logging.error(f'Some error occurred {e}')




def train_and_register():

    df = load_data('./data/interim/train_preprocessed.csv')

    categorical_cols = ['Company','TypeName','Cpu brand','Gpu Brand','os']

    X = df.drop(columns='Price')
    y = np.log(df['Price'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": 100,
        "max_depth": 25,
        "max_features": 0.7,
        "max_samples": 0.75
    }

    # Set experiment
    mlflow.set_experiment("laptop_price_prediction")
    model_name = "plmodel"

    with mlflow.start_run() as run:

        pipeline = build_pipeline(params, categorical_cols)

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
        }
        

        # Log params
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)
        

        # Log & register model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="plmodel"   
        )

        client = MlflowClient()

    
        client = MlflowClient()

        latest_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        latest_version = sorted(
            latest_versions,
            key=lambda x: int(x.version)
        )[-1].version


        client.set_registered_model_alias(
            name=model_name,
            alias="Staging",
            version=latest_version
        )

        client.set_registered_model_tag(
            name=model_name,
            key="project",
            value="laptop-price-prediction"
        )
        save_pipeline(pipeline,'./models/pipeline.pkl')
        save_metrics(metrics,'./reports/metrics.json')

       
if __name__ == "__main__":
    train_and_register()