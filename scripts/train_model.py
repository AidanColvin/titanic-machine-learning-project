import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_parquet(filepath: str) -> pd.DataFrame:
    """
    given a filepath string
    return dataframe 
    loaded from parquet format
    """
    return pd.read_parquet(filepath)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe 
    return dataframe ready for modeling 
    text columns dropped 
    categorical columns one-hot encoded
    """
    df = df.copy()
    
    # Drop high-cardinality text columns/noise
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    # Only drop if they exist
    existing_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop)
    
    # One-Hot Encode categorical variables (Title, Embarked, etc.)
    # pd.get_dummies converts strings to binary columns (0/1)
    df = pd.get_dummies(df, columns=['Embarked', 'Title'], drop_first=True)
    
    return df

def align_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    given train and test dataframes 
    return aligned dataframes 
    ensures both have identical columns 
    missing columns filled with 0
    """
    # Get all columns from both, essentially the union of features
    train_cols = train_df.columns
    test_cols = test_df.columns
    
    # Align test to train (Machine Learning models need exact same input structure)
    # This adds missing 'Title_...' columns to test if they didn't exist there
    for col in train_cols:
        if col not in test_cols:
            test_df[col] = 0
            
    # Remove columns in test that aren't in train
    test_df = test_df[train_cols]
    
    return train_df, test_df

if __name__ == "__main__":
    print("\n[Step 4] Training Model...")
    
    # 1. Load Data
    try:
        train_raw = load_parquet("data/train_processed.parquet")
        test_raw = load_parquet("data/test_processed.parquet")
    except FileNotFoundError:
        print("Error: Processed data not found. Run scripts/process_data.py first.")
        exit()

    # 2. Separate Target (y) from Features (X)
    y = train_raw['Survived']
    X_raw = train_raw.drop(columns=['Survived'])
    
    # 3. Final Preprocessing (Encoding)
    print("Encoding categorical features...")
    X = prepare_features(X_raw)
    X_test = prepare_features(test_raw)
    
    # 4. Align Columns (Critical Step)
    # One-Hot Encoding might create different columns if test data is missing a category
    X, X_test = align_columns(X, X_test)
    
    # 5. Split Training Data for Validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Train Model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X_train, y_train)
    
    # 7. Evaluate
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # 8. Generate Submission
    final_predictions = model.predict(X_test)
    
    # We need PassengerId for submission, retrieving from original file
    submission = pd.read_csv("data/gender_submission.csv")
    submission['Survived'] = final_predictions
    
    submission.to_csv("data/submission.csv", index=False)
    print("Success: Submission saved to 'data/submission.csv'")
