import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_raw_data(data_path=os.path.join("data", "raw", "insurance.csv")):
    """
    Loads raw medical insurance data.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at {data_path}")
    return pd.read_csv(data_path)

def clean_data(df):
    """
    Cleans raw dataframe:
    - Removes duplicates
    - Standardizes data types
    - Checks for and handles missing values
    """
    cleaned_df = df.copy()
    
    # 1. Handle Duplicates
    initial_shape = cleaned_df.shape
    cleaned_df = cleaned_df.drop_duplicates()
    duplicates_removed = initial_shape[0] - cleaned_df.shape[0]
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows.")
        
    # 2. Check for missing values
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: Found {missing_count} missing values. Handling them...")
        # Since missing values are not standard in this dataset, we can drop them or impute them.
        cleaned_df = cleaned_df.dropna()
        
    # 3. Convert types and strip whitespaces in string columns
    cleaned_df['sex'] = cleaned_df['sex'].astype(str).str.strip().str.lower()
    cleaned_df['smoker'] = cleaned_df['smoker'].astype(str).str.strip().str.lower()
    cleaned_df['region'] = cleaned_df['region'].astype(str).str.strip().str.lower()
    
    # Convert numerical types explicitly
    cleaned_df['age'] = cleaned_df['age'].astype(int)
    cleaned_df['bmi'] = cleaned_df['bmi'].astype(float)
    cleaned_df['children'] = cleaned_df['children'].astype(int)
    cleaned_df['charges'] = cleaned_df['charges'].astype(float)
    
    return cleaned_df

def build_preprocessor():
    """
    Builds a scikit-learn ColumnTransformer for preprocessing
    numerical and categorical columns.
    """
    numeric_features = ['age', 'bmi', 'children']
    categorical_features = ['sex', 'smoker', 'region']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def run_preprocessing_pipeline():
    """
    Runs the full loading, cleaning, and preprocessing steps.
    Saves the cleaned dataset.
    """
    print("Starting data preprocessing pipeline...")
    # Load
    raw_df = load_raw_data()
    # Clean
    cleaned_df = clean_data(raw_df)
    
    # Save cleaned data for verification and EDA
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    cleaned_path = os.path.join(processed_dir, "cleaned_insurance.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Cleaned dataset saved successfully to {cleaned_path}")
    
    return cleaned_df

if __name__ == "__main__":
    run_preprocessing_pipeline()
