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
    if not isinstance(name, str): return ""
    search = re.search(' ([A-Za-z]+)\.', name)
    if search:
        return search.group(1)
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
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Misc')
        df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe 
    return processed dataframe 
    titles extracted 
    family features created
    """
    df = simplify_titles(df)
    df = create_family_size(df)
    df = create_is_alone(df)
    return df

if __name__ == "__main__":
    files = ["train", "test"]
    
    print("\n[Step 3] Processing Features...")
    for name in files:
        input_path = f"data/{name}_cleaned.parquet"
        output_path = f"data/{name}_processed.parquet"
        
        try:
            df = load_parquet(input_path)
            df_proc = process_dataset(df)
            df_proc.to_parquet(output_path)
            print(f"processed {input_path} saved to {output_path}")
        except FileNotFoundError:
            print(f"skipping {name}: input file not found")
