import pandas as pd
import re
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    given a filepath string
    return dataframe loaded from parquet
    """
    return pd.read_parquet(filepath)

def extract_title(name: str) -> str:
    """
    given a name string 
    return the title 
    found between comma and period
    """
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe 
    return dataframe with FamilySize column 
    sum of SibSp and Parch plus one
    """
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df

def create_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe 
    return dataframe with IsAlone column 
    1 if FamilySize is 1, else 0
    """
    df = df.copy()
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    return df

def simplify_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe 
    return dataframe with simplified Title column 
    rare titles grouped as 'Misc'
    """
    df = df.copy()
    if 'Name' in df.columns:
        df['Title'] = df['Name'].apply(extract_title)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Misc')
        df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe 
    return fully processed dataframe 
    titles extracted and features engineered
    """
    df = simplify_titles(df)
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df = create_family_size(df)
        df = create_is_alone(df)
    return df

if __name__ == "__main__":
    # Define paths assuming data is in 'data' folder
    files = {
        'train': 'data/train_cleaned.parquet',
        'test': 'data/test_cleaned.parquet'
    }
    
    print("Starting advanced feature engineering...")
    
    for name, filepath in files.items():
        if os.path.exists(filepath):
            df = load_data(filepath)
            processed_df = process_dataset(df)
            
            output_path = filepath.replace("_cleaned.parquet", "_processed.parquet")
            processed_df.to_parquet(output_path)
            print(f"Processed {name} data saved to {output_path}")
        else:
            print(f"Warning: {filepath} not found. Did you run clean_data.py?")
            
    print("Processing complete.")
