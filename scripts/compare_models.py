import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data():
    df = pd.read_parquet("data/train_processed.parquet")
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = pd.get_dummies(df, columns=['Embarked', 'Title'], drop_first=True)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    return X, y

if __name__ == "__main__":
    print("\n" + "="*40)
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