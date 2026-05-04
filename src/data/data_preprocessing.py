import numpy as np
import pandas as pd
from pandas import DataFrame
from src.logger import logging
import os

def preprocess(df: DataFrame) -> pd.DataFrame:
    try:
        logging.info("Data Preprocessing Started")

        # ---- Basic Cleaning ----
        df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
        df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float32')

        # ---- Screen Features ----
        df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False, na=False).astype(int)
        df['IPS'] = df['ScreenResolution'].str.contains('IPS', case=False, na=False).astype(int)

        # ---- Resolution ----
        new = df['ScreenResolution'].str.split('x', n=1, expand=True)

        df['x_resolution'] = (
            new[0]
            .str.replace(',', '', regex=False)
            .str.extract(r'(\d+\.?\d+)')[0]
            .astype(float)
        )

        df['y_resolution'] = pd.to_numeric(new[1], errors='coerce')

        # ---- PPI ----
        df['PPI'] = ((df['x_resolution']**2 + df['y_resolution']**2)**0.5) / df['Inches']

        df.drop(columns=['ScreenResolution', 'x_resolution', 'y_resolution', 'Inches'], inplace=True)

        # ---- CPU ----
        df['Cpu_Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[:3]))

        def fetch_processor(text):
            if text in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
                return text
            elif text.startswith('Intel'):
                return 'Other Intel Processor'
            else:
                return 'AMD Processor'

        df['Cpu brand'] = df['Cpu_Name'].apply(fetch_processor)
        df.drop(columns=['Cpu', 'Cpu_Name'], inplace=True)

        # ---- Memory ----
        df['Memory'] = df['Memory'].astype(str)
        df['Memory'] = df['Memory'].str.replace('\.0', '', regex=True)
        df['Memory'] = df['Memory'].str.replace('GB', '', regex=False)
        df['Memory'] = df['Memory'].str.replace('TB', '000', regex=False)

        new = df['Memory'].str.split('+', n=1, expand=True)

        df['First'] = new[0].str.strip()
        df['Second'] = new[1].fillna('0').str.strip()

        # ---- Storage Flags ----
        for col, name in [('First', 'Layer1'), ('Second', 'Layer2')]:
            df[f'{name}SSD'] = df[col].str.contains('SSD', case=False, na=False).astype(int)
            df[f'{name}HDD'] = df[col].str.contains('HDD', case=False, na=False).astype(int)
            df[f'{name}Hybrid'] = df[col].str.contains('Hybrid', case=False, na=False).astype(int)
            df[f'{name}Flash_Storage'] = df[col].str.contains('Flash Storage', case=False, na=False).astype(int)

        # ---- Extract numbers ----
        df['First'] = pd.to_numeric(df['First'].str.replace(r'\D', '', regex=True), errors='coerce').fillna(0)
        df['Second'] = pd.to_numeric(df['Second'].str.replace(r'\D', '', regex=True), errors='coerce').fillna(0)

        # ---- Final Storage ----
        df['HDD'] = df['First'] * df['Layer1HDD'] + df['Second'] * df['Layer2HDD']
        df['SSD'] = df['First'] * df['Layer1SSD'] + df['Second'] * df['Layer2SSD']

        # ---- Cleanup ----
        df.drop(columns=[
            'First','Second',
            'Layer1SSD','Layer1HDD','Layer1Hybrid','Layer1Flash_Storage',
            'Layer2SSD','Layer2HDD','Layer2Hybrid','Layer2Flash_Storage',
            'Memory','Hybrid','Flash_Storage'
        ], inplace=True, errors='ignore')

        # ---- GPU ----
        df['Gpu Brand'] = df['Gpu'].apply(lambda x: x.split()[0] if pd.notna(x) else 'Unknown')
        df = df[df['Gpu Brand'] != 'ARM']
        df.drop(columns=['Gpu'], inplace=True)

        # ---- OS ----
        def cat_os(inp):
            inp = str(inp).strip()
            if inp in ['Windows 10', 'Windows 7', 'Windows 10 S']:
                return 'Windows'
            elif inp in ['macOS', 'Mac OS X']:
                return 'Mac'
            else:
                return 'Others/No os/Linux'

        df['os'] = df['OpSys'].apply(cat_os)
        df.drop(columns=['OpSys'], inplace=True)

        logging.info("Data Preprocessing Completed")
        return df

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise


def main():
    try:
        train_data = pd.read_csv('./data/raw/train_data.csv')
        test_data = pd.read_csv('./data/raw/test_data.csv')
        logging.info('Trained data loaded successfully:')

        train_preprocessed_data = preprocess(train_data)
        test_preprocessed_data = preprocess(test_data)
        logging.info('Successfully preprocessed data:')

        data_path = os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)

        train_preprocessed_data.to_csv(os.path.join(data_path,'train_preprocessed.csv'),index=False)
        test_preprocessed_data.to_csv(os.path.join(data_path,'test_preprocessed.csv'),index=False)
        logging.info(f'Data saved successfully in {data_path}')

    except Exception as e:
        logging.error('Some error occurred %s',e)

if __name__ == '__main__':
    main()
