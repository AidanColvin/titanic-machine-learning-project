import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Suppress warnings
warnings.filterwarnings('ignore')

def load_and_prep_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads data and applies the winning Feature Engineering strategy:
    TicketFreq + TitleGroup + FarePerPerson + Deck + IsMother
    Returns: X, y, X_test
    """
    try:
        train = pd.read_csv("data/train.csv")
        test = pd.read_csv("data/test.csv")
    except FileNotFoundError:
        print("Error: Files not found.")
        exit()

    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], sort=False).reset_index(drop=True)
    
    # --- 1. Title Extraction & Mapping ---
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
        'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1,
        'Countess': 5, 'Ms': 1, 'Lady': 5, 'Jonkheer': 4,
        'Don': 4, 'Dona': 5, 'Mme': 2, 'Capt': 4, 'Sir': 4
    }
    df['TitleGroup'] = df['Title'].map(title_mapping).fillna(4)

    # --- 2. Advanced Features ---
    # Ticket Frequency (The #1 Predictor for 86%)
    df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')
    
    # Fare Per Person (Normalizes price)
    df['FarePerPerson'] = df['Fare'] / df['TicketFreq']
    
    # Deck (Cabin Letter)
    df['Deck'] = df['Cabin'].str[0] if 'Cabin' in df.columns else 'U'
    deck_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'U': 8}
    df['Deck'] = df['Deck'].map(deck_mapping).fillna(8)
    
    # Family Size & Mother
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsMother'] = ((df['Sex'] == 'female') & (df['Parch'] > 0) & (df['Age'] > 18)).astype(int)
    
    # --- 3. Clean Up ---
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Title']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Encode Sex/Embarked
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Impute Missing Values
    imputer = SimpleImputer(strategy='median')
    cols = df.columns
    df = pd.DataFrame(imputer.fit_transform(df), columns=cols)
    
    # Split
    X = df[df['is_train'] == 1].drop(columns=['is_train', 'Survived'])
    y = df[df['is_train'] == 1]['Survived']
    X_test = df[df['is_train'] == 0].drop(columns=['is_train', 'Survived'])
    
    return X, y, X_test

def robust_evaluate(model, X, y, n_splits=5) -> tuple[float, float]:
    """
    Manually performs Stratified K-Fold Cross Validation.
    This bypasses the sklearn/CatBoost compatibility bug.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []
    
    # Convert to numpy for easier indexing
    X_np = X.values
    y_np = y.values
    
    for train_idx, val_idx in skf.split(X_np, y_np):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
        
    return np.mean(scores), np.std(scores)

if __name__ == "__main__":
    X, y, X_test = load_and_prep_data()
    
    print("\n" + "="*55)
    print("      TITANIC: THE FINAL 'TITANS' BENCHMARK")
    print("="*55)
    print("Strategy: Low Learning Rate + Weighted Soft Voting")
    print("-" * 55)
    
    # --- 1. Define The Titans (Slow Learners) ---
    
    # XGBoost: Deep trees, slow learning
    xgb = XGBClassifier(
        n_estimators=1000, learning_rate=0.01, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8, random_state=1, 
        eval_metric='logloss', n_jobs=-1
    )
    
    # LightGBM: High leaf count
    lgbm = LGBMClassifier(
        n_estimators=1000, learning_rate=0.01, max_depth=5, 
        num_leaves=31, random_state=1, verbose=-1, n_jobs=-1
    )
    
    # CatBoost: Depth 6 is optimal for Titanic
    cat = CatBoostClassifier(
        iterations=1000, learning_rate=0.01, depth=6, 
        verbose=0, random_state=1, allow_writing_files=False
    )
    
    # Random Forest: Diversity injector
    rf = RandomForestClassifier(n_estimators=500, max_depth=7, random_state=1, n_jobs=-1)
    
    # --- 2. Weighted Voting Ensemble ---
    # Weight: 3 for Boosters, 1 for RF (Boosters are significantly more accurate here)
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('cat', cat),
            ('rf', rf)
        ],
        voting='soft',
        weights=[3, 3, 3, 1]
    )
    
    models = [
        ("Random Forest", rf),
        ("XGBoost", xgb),
        ("LightGBM", lgbm),
        ("CatBoost", cat),
        (">> WEIGHTED ENSEMBLE <<", ensemble)
    ]
    
    # --- 3. Robust Benchmark ---
    print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Std Dev':<10}")
    print("-" * 55)
    
    for name, model in models:
        acc, std = robust_evaluate(model, X, y)
        print(f"{name:<25} | {acc:.4f}     | {std:.4f}")
    
    print("-" * 55)
    
    # --- 4. Final Submission ---
    ensemble.fit(X, y)
    preds = ensemble.predict(X_test).astype(int)
    
    sub = pd.read_csv("data/gender_submission.csv")
    sub['Survived'] = preds
    sub.to_csv("data/submission_final.csv", index=False)
    print("Success: Final submission saved to 'data/submission_final.csv'")
