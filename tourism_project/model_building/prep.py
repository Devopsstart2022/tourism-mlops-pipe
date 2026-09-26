
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def prepare_data(input_path='tourism_project/data/tourism.csv',
                 output_dir='tourism_project/artifacts'):

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(input_path, index_col=0)
    print(f'Loaded data: {df.shape}')

    # Drop CustomerID — unique identifier, no predictive value
    df.drop(columns=['CustomerID'], errors='ignore', inplace=True)

    # Impute missing values
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in num_cols:
        if col != 'ProdTaken':
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    print(f'Missing after imputation: {df.isnull().sum().sum()}')

    # Encode categorical variables
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']

    # 80/20 train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save splits
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv',  index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv',   index=False)

    print(f'Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}')
    print('Train/test splits saved to tourism_project/artifacts/')
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    prepare_data()
