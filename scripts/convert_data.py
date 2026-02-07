import pandas as pd
import os

def load_csv(filepath: str) -> pd.DataFrame:
    """
    given a filepath string
    return dataframe
    loaded from csv format
    """
    return pd.read_csv(filepath)

def save_parquet(df: pd.DataFrame, filepath: str) -> None:
    """
    given a dataframe and filepath
    save dataframe as parquet
    snappy compression used
    """
    df.to_parquet(filepath, engine='pyarrow', compression='snappy')

if __name__ == "__main__":
    files = ["train", "test", "gender_submission"]
    
    print("\n[Step 1] Converting CSV to Parquet...")
    for name in files:
        csv_path = f"data/{name}.csv"
        parquet_path = f"data/{name}.parquet"
        
        if os.path.exists(csv_path):
            df = load_csv(csv_path)
            save_parquet(df, parquet_path)
            print(f"converted {csv_path} to {parquet_path}")
        else:
            print(f"warning: {csv_path} not found")
