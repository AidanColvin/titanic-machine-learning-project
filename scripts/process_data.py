import pandas as pd
import re

def load_parquet(filepath: str) -> pd.DataFrame:
    """
    given a filepath string
    return dataframe 
    loaded from parquet format
    """
    return pd.read_parquet(filepath)

def extract_title(name: str) -> str:
    """
    given a name string 
    return the title 
    found between comma and period
    """
    # Regex searches for space, letters, then a dot
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
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df

def create_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe 
    return dataframe with IsAlone column 
    1 if FamilySize is 1, else 0
    """
    df = df.copy()
    if 'FamilySize' in df.columns:
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
        
        rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df['Title'] = df['Title'].replace(rare_titles, 'Misc')
        
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

def final_preprocessing(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    given train and test dataframes 
    return both dataframes fully processed 
    titles extracted and features engineered
    """
    # Combine for structural consistency as requested
    combined = [train_df, test_df]
    processed = []
    
    for df in combined:
        df = simplify_titles(df)
        df = create_family_size(df)
        df = create_is_alone(df)
        processed.append(df)
        
    return processed[0], processed[1]

if __name__ == "__main__":
    print("\n[Step 3] Processing Features (Feature Engineering)...")
    
    try:
        # Load cleaned data
        train = load_parquet("data/train_cleaned.parquet")
        test = load_parquet("data/test_cleaned.parquet")
        
        # Apply final preprocessing logic
        train_processed, test_processed = final_preprocessing(train, test)
        
        # Save results
        train_processed.to_parquet("data/train_processed.parquet")
        test_processed.to_parquet("data/test_processed.parquet")
        
        print("processed data/train_cleaned.parquet saved to data/train_processed.parquet")
        print("processed data/test_cleaned.parquet saved to data/test_processed.parquet")
        
    except FileNotFoundError:
        print("error: cleaned files not found. run clean_data.py first.")
