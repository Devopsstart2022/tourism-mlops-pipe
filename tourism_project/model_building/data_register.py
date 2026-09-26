import pandas as pd
import sys

EXPECTED_COLUMNS = [
    'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
    'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]

def register_dataset(path='tourism_project/data/tourism.csv'):
    print('=' * 55)
    print('DATA REGISTRATION & VALIDATION')
    print('=' * 55)
    df = pd.read_csv(path, index_col=0)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print('VALIDATION FAILED — Missing columns:', missing_cols)
        sys.exit(1)
    print('Column validation: PASSED')
    print('Rows       :', df.shape[0])
    print('Columns    :', df.shape[1])
    print('Target (ProdTaken) distribution:')
    print(df['ProdTaken'].value_counts().to_string())
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print('Missing values:')
    print(missing.to_string() if len(missing) > 0 else 'None')
    print('Data registration complete.')
    return df

if __name__ == '__main__':
    register_dataset()
