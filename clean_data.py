import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    given a filepath string
    return dataframe 
    loaded from parquet file
    """
    return pd.read_parquet(filepath)

def fill_ages(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with missing ages filled
    filled with median age if column exists
    """
    df = df.copy()
    if 'Age' in df.columns:
        median_age = df['Age'].median()
        df['Age'] = df['Age'].fillna(median_age)
    return df

def fill_fare(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with missing fare filled
    filled with median fare if column exists
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
    filled with mode value if column exists
    """
    df = df.copy()
    if 'Embarked' in df.columns:
        mode_embarked = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(mode_embarked)
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

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with noisy columns dropped
    name and ticket removed
    cabin removed
    """
    df = df.copy()
    cols = ['Name', 'Ticket', 'Cabin']
    existing_cols = [c for c in cols if c in df.columns]
    return df.drop(columns=existing_cols)

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a raw dataframe
    return fully cleaned dataframe
    missing values filled if columns exist
    noise removed
    """
    df = fill_ages(df)
    df = fill_fare(df)
    df = fill_embarked(df)
    df = encode_gender(df)
    return drop_unnecessary_columns(df)

if __name__ == "__main__":
    # List of files to process in the repository
    target_files = [
        "train.parquet", 
        "test.parquet", 
        "gender_submission.parquet"
    ]
    
    print("Starting data cleaning process...\n")
    
    for file in target_files:
        try:
            # Load and process the data
            raw_data = load_data(file)
            cleaned_data = clean_dataset(raw_data)
            
            # Generate the new filename
            output_name = file.replace(".parquet", "_cleaned.parquet")
            
            # Save the new version back to the repository
            cleaned_data.to_parquet(output_name)
            
            # Print confirmation to terminal
            print(f"cleaned {file} and saved to {output_name}")
            
        except Exception as e:
            # Report any errors specifically for that file
            print(f"failed to process {file}: {e}")

    print("\nAll tasks completed.")