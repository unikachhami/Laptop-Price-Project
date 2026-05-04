from src.logger import logging
import yaml
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_params(params_path: str)->dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('parameters safely loaded from path %d',params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found %d',params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('Yaml error %s',e)
    except Exception as e:
        logging.error('File not found %s',e)

def load_data(file_path: str)->pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.debug('Data safely loaded from %d',file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse csv file %s',e)
        raise

    except Exception as e:
        logging.error('File not found %s',e)
        raise

def preprocess_df(df: pd.DataFrame)-> pd.DataFrame:
    try:
        logging.info('Pre-processing started:')
        final_df = df.drop(columns=['Unnamed: 0'])
        logging.info('Pre-processing completed:')
        return final_df
    except Exception as e:
        logging.error('Failed to make changes %s',e)
        raise

def save_df(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path)-> pd.DataFrame:
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_df.to_csv(os.path.join(raw_data_path,'train_data.csv'),index=False)
        test_df.to_csv(os.path.join(raw_data_path,'test_data.csv'),index=False)
        logging.info('Data Successfully saved to %s',data_path)

    except Exception as e:
        logging.error('Failed to save csv %s',e)
        raise

def main():
    try:
        df = load_data('https://raw.githubusercontent.com/unikachhami/Laptop-Price-Project/refs/heads/main/laptop_data.csv')
        # obj = s3_connection.s3_operations(bucket_name='bucket_name',aws_access_key='aws_access_key',aws_secret_key='aws_secret_key')
        # df= obj.fetch_file_from_s3(file_key=file_key)

        final_df = preprocess_df(df)
        train_df,test_df =train_test_split(final_df,test_size=0.2,random_state=42)
        save_df(train_df,test_df,data_path='./data')
    except Exception as e:
        logging.error("Some error occurred %s",e)


if __name__ == "__main__":
    main()