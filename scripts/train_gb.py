import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def load_data():
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
    print("\n--- Training Gradient Boosting ---")
    try:
        X, y, X_test = load_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"Validation Accuracy: {acc:.4f}")
        preds = model.predict(X_test)
        sub = pd.read_csv("data/gender_submission.csv")
        sub['Survived'] = preds
        sub.to_csv("data/submission_gb.csv", index=False)
        print("Saved: data/submission_gb.csv")
    except Exception as e:
        print(f"Error: {e}")