import os
import textwrap

def write_file(path: str, content: str) -> None:
    """
    writes content to a file
    creates parent directories if needed
    """
    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Write the file
    with open(path, 'w') as f:
        f.write(content.strip())
    print(f"✅ Created: {path}")

def main() -> None:
    """
    generates all project files
    organizes folders
    """
    print(">>> Starting Project Setup...\n")

    # --- 1. Random Forest Script ---
    rf_code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    \"\"\"
    loads processed data
    returns X (features), y (target), and X_test (submission data)
    \"\"\"
    train_df = pd.read_parquet("data/train_processed.parquet")
    test_df = pd.read_parquet("data/test_processed.parquet")
    
    # Drop non-numeric/high-cardinality columns
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    
    # One-Hot Encoding
    train_df = pd.get_dummies(train_df, columns=['Embarked', 'Title'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['Embarked', 'Title'], drop_first=True)
    
    # Align Columns
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for c in missing_cols:
        if c != 'Survived':
            test_df[c] = 0
    test_df = test_df[train_df.drop(columns=['Survived']).columns]
    
    y = train_df['Survived']
    X = train_df.drop(columns=['Survived'])
    
    return X, y, test_df

if __name__ == "__main__":
    print("\\n--- Training Random Forest ---")
    try:
        X, y, X_test = load_data()
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"Random Forest Validation Accuracy: {acc:.4f}")
        
        preds = model.predict(X_test)
        sub = pd.read_csv("data/gender_submission.csv")
        sub['Survived'] = preds
        sub.to_csv("data/submission_rf.csv", index=False)
        print("Saved: data/submission_rf.csv")
    except Exception as e:
        print(f"Error: {e}")
"""
    write_file("scripts/train_rf.py", rf_code)

    # --- 2. Logistic Regression Script ---
    logreg_code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    \"\"\"
    loads processed data
    returns X, y, X_test
    \"\"\"
    train_df = pd.read_parquet("data/train_processed.parquet")
    test_df = pd.read_parquet("data/test_processed.parquet")
    
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    
    train_df = pd.get_dummies(train_df, columns=['Embarked', 'Title'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['Embarked', 'Title'], drop_first=True)
    
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for c in missing_cols:
        if c != 'Survived':
            test_df[c] = 0
    test_df = test_df[train_df.drop(columns=['Survived']).columns]
    
    y = train_df['Survived']
    X = train_df.drop(columns=['Survived'])
    return X, y, test_df

if __name__ == "__main__":
    print("\\n--- Training Logistic Regression ---")
    try:
        X, y, X_test = load_data()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)
        
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000, random_state=1)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"Logistic Regression Validation Accuracy: {acc:.4f}")
        
        preds = model.predict(X_test_scaled)
        sub = pd.read_csv("data/gender_submission.csv")
        sub['Survived'] = preds
        sub.to_csv("data/submission_logreg.csv", index=False)
        print("Saved: data/submission_logreg.csv")
    except Exception as e:
        print(f"Error: {e}")
"""
    write_file("scripts/train_logreg.py", logreg_code)

    # --- 3. Gradient Boosting Script ---
    gb_code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    \"\"\"
    loads processed data
    returns X, y, X_test
    \"\"\"
    train_df = pd.read_parquet("data/train_processed.parquet")
    test_df = pd.read_parquet("data/test_processed.parquet")
    
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    
    train_df = pd.get_dummies(train_df, columns=['Embarked', 'Title'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['Embarked', 'Title'], drop_first=True)
    
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for c in missing_cols:
        if c != 'Survived':
            test_df[c] = 0
    test_df = test_df[train_df.drop(columns=['Survived']).columns]
    
    y = train_df['Survived']
    X = train_df.drop(columns=['Survived'])
    return X, y, test_df

if __name__ == "__main__":
    print("\\n--- Training Gradient Boosting ---")
    try:
        X, y, X_test = load_data()
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"Gradient Boosting Validation Accuracy: {acc:.4f}")
        
        preds = model.predict(X_test)
        sub = pd.read_csv("data/gender_submission.csv")
        sub['Survived'] = preds
        sub.to_csv("data/submission_gb.csv", index=False)
        print("Saved: data/submission_gb.csv")
    except Exception as e:
        print(f"Error: {e}")
"""
    write_file("scripts/train_gb.py", gb_code)

    # --- 4. SVM Script ---
    svm_code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    \"\"\"
    loads processed data
    returns X, y, X_test
    \"\"\"
    train_df = pd.read_parquet("data/train_processed.parquet")
    test_df = pd.read_parquet("data/test_processed.parquet")
    
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    
    train_df = pd.get_dummies(train_df, columns=['Embarked', 'Title'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['Embarked', 'Title'], drop_first=True)
    
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for c in missing_cols:
        if c != 'Survived':
            test_df[c] = 0
    test_df = test_df[train_df.drop(columns=['Survived']).columns]
    
    y = train_df['Survived']
    X = train_df.drop(columns=['Survived'])
    return X, y, test_df

if __name__ == "__main__":
    print("\\n--- Training SVM ---")
    try:
        X, y, X_test = load_data()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)
        
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = SVC(kernel='rbf', C=1.0, random_state=1)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"SVM Validation Accuracy: {acc:.4f}")
        
        preds = model.predict(X_test_scaled)
        sub = pd.read_csv("data/gender_submission.csv")
        sub['Survived'] = preds
        sub.to_csv("data/submission_svm.csv", index=False)
        print("Saved: data/submission_svm.csv")
    except Exception as e:
        print(f"Error: {e}")
"""
    write_file("scripts/train_svm.py", svm_code)

    # --- 5. Comparison Script ---
    compare_code = """
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    \"\"\"
    loads data for comparison
    returns X, y
    \"\"\"
    df = pd.read_parquet("data/train_processed.parquet")
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = pd.get_dummies(df, columns=['Embarked', 'Title'], drop_first=True)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    return X, y

if __name__ == "__main__":
    print("\\n" + "="*40)
    print("      MODEL COMPARISON LEADERBOARD")
    print("="*40)
    
    try:
        X, y = load_data()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        models = [
            ("Gradient Boosting", GradientBoostingClassifier(random_state=1), X),
            ("Random Forest    ", RandomForestClassifier(random_state=1), X),
            ("Support Vector   ", SVC(random_state=1), X_scaled),
            ("Logistic Regress ", LogisticRegression(max_iter=1000), X_scaled)
        ]
        
        print(f"{'Model Name':<20} | {'Accuracy (Mean CV)':<20}")
        print("-" * 45)
        
        for name, model, data in models:
            scores = cross_val_score(model, data, y, cv=5, scoring='accuracy')
            print(f"{name:<20} | {np.mean(scores):.4f}")
        print("-" * 45)
    except Exception as e:
        print(f"Error: {e}")
"""
    write_file("scripts/compare_models.py", compare_code)

    # --- 6. Documentation ---
    doc_code = """
# Titanic Model Experiments & Comparison

## Overview
This project evaluates four distinct machine learning algorithms to determine which best captures survival patterns.

## 1. Models Implemented
* **Random Forest** (`scripts/train_rf.py`): Baseline ensemble model.
* **Logistic Regression** (`scripts/train_logreg.py`): Linear baseline, highly interpretable.
* **Gradient Boosting** (`scripts/train_gb.py`): Sequential trees, often highest accuracy.
* **SVM** (`scripts/train_svm.py`): Finds optimal class separation boundary.

## 2. Usage
Run the comparison leaderboard:
`python scripts/compare_models.py`

Generate individual submissions:
`python scripts/train_gb.py`
"""
    write_file("docs/model_experiments.md", doc_code)

    print("\n>>> Success! All files generated.")
    print(">>> Run the comparison now using: python scripts/compare_models.py")

if __name__ == "__main__":
    main()