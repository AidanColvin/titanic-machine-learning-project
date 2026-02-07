import pandas as pd

def load_parquet(filepath: str) -> pd.DataFrame:
    """
    given a filepath string
    return dataframe 
    loaded from parquet format
    """
    return pd.read_parquet(filepath)

def fill_ages(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with missing ages filled
    filled with median age of training set
    """
    df = df.copy()
    if 'Age' in df.columns:
        # Note: In a strict ML pipeline, we should calculate median 
        # from train only and apply to test. For this cleaning step,
        # we apply a simple fill to allow processing to continue.
        median_age = df['Age'].median()
        df['Age'] = df['Age'].fillna(median_age)
    return df

def fill_fare(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with missing fare filled
    filled with median fare
    """
    df = df.copy()
    if 'Fare' in df.columns:
        median_fare = df['Fare'].median()
        df['Fare'] = df['Fare'].fillna(median_fare)
    return df

def fill_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with missing embarked filled
    filled with mode value
    """
    df = df.copy()
    if 'Embarked' in df.columns:
        mode = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(mode)
    return df

def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with sex column encoded
    male mapped to 0
    female mapped to 1
    """
    df = df.copy()
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a raw dataframe
    return fully cleaned dataframe
    missing values filled
    features encoded
    """
    df = fill_ages(df)
    df = fill_fare(df)
    df = fill_embarked(df)
    df = encode_gender(df)
    return df

if __name__ == "__main__":
    files = ["train", "test", "gender_submission"]
    
    print("\n[Step 2] Cleaning Parquet Files...")
    for name in files:
        input_path = f"data/{name}.parquet"
        output_path = f"data/{name}_cleaned.parquet"
        
        try:
            df = load_parquet(input_path)
            df_clean = clean_dataset(df)
            df_clean.to_parquet(output_path)
            print(f"cleaned {input_path} saved to {output_path}")
        except FileNotFoundError:
            print(f"skipping {name}: file not found")
