import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# --- FIX: Custom Wrapper to make CatBoost compatible with new Scikit-Learn ---
class SklearnCompatibleCatBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations=1000, learning_rate=0.01, depth=6, random_state=1):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state
        self.model = CatBoostClassifier(
            iterations=iterations, 
            learning_rate=learning_rate, 
            depth=depth, 
            random_state=random_state,
            verbose=0,
            allow_writing_files=False
        )

    def fit(self, X, y):
        """
        given X and y
        fit the internal catboost model
        return self
        """
        self.model.fit(X, y)
        # Required for VotingClassifier to know class labels
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        """
        given X
        return predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        given X
        return probability estimates
        """
        return self.model.predict_proba(X)
    
    def __sklearn_tags__(self):
        """
        bypass for sklearn 1.6+ compatibility check
        """
        from sklearn.utils._tags import ClassifierTags
        return ClassifierTags()

def load_and_prep_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    given raw csv files
    return prepared X, y, and X_test
    includes ticket frequency, title grouping, and fare per person
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
    # Ticket Frequency (The #1 Predictor)
    df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')
    df['FarePerPerson'] = df['Fare'] / df['TicketFreq']
    
    # Deck
    df['Deck'] = df['Cabin'].str[0] if 'Cabin' in df.columns else 'U'
    deck_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'U': 8}
    df['Deck'] = df['Deck'].map(deck_mapping).fillna(8)
    
    # Family
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsMother'] = ((df['Sex'] == 'female') & (df['Parch'] > 0) & (df['Age'] > 18)).astype(int)
    
    # --- 3. Clean Up ---
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Title']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Impute
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
    given model and data
    return mean accuracy and std dev
    uses manual stratified k-fold to avoid sklearn version conflicts
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []
    X_np, y_np = X.values, y.values
    
    for train_idx, val_idx in skf.split(X_np, y_np):
        model.fit(X_np[train_idx], y_np[train_idx])
        preds = model.predict(X_np[val_idx])
        scores.append(accuracy_score(y_np[val_idx], preds))
        
    return np.mean(scores), np.std(scores)

if __name__ == "__main__":
    X, y, X_test = load_and_prep_data()
    
    print("\n" + "="*60)
    print("      TITANIC ULTIMATE LEADERBOARD (ALL MODELS)")
    print("="*60)
    print(f"{'Model Name':<30} | {'Accuracy':<10} | {'Std Dev':<10}")
    print("-" * 60)
    
    # --- DEFINE ALL MODELS (Slow Learning Strategy) ---
    
    # 1. Baselines
    rf = RandomForestClassifier(n_estimators=500, max_depth=7, random_state=1, n_jobs=-1)
    lr = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=1000))])
    svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, kernel='rbf', random_state=1))])
    
    # 2. The Titans (Boosters) - High trees, low learning rate
    xgb = XGBClassifier(
        n_estimators=1000, learning_rate=0.01, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8, random_state=1, 
        eval_metric='logloss', n_jobs=-1
    )
    lgbm = LGBMClassifier(
        n_estimators=1000, learning_rate=0.01, max_depth=5, 
        num_leaves=31, random_state=1, verbose=-1, n_jobs=-1
    )
    
    # 3. CatBoost (Wrapped)
    cat = SklearnCompatibleCatBoost(iterations=1000, learning_rate=0.01, depth=6, random_state=1)
    
    # 4. Ensembles
    # Stacking: Uses LogReg to learn when to trust which booster
    stacking = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('cat', cat), ('lgbm', lgbm)],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )
    
    # Voting: Weighted soft voting (Boosters > RF)
    voting = VotingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat), ('rf', rf)],
        voting='soft',
        weights=[3, 3, 3, 1],
        n_jobs=-1
    )
    
    all_models = [
        ("Logistic Regression", lr),
        ("Support Vector Machine", svm),
        ("Random Forest", rf),
        ("XGBoost", xgb),
        ("LightGBM", lgbm),
        ("CatBoost", cat),
        (">> STACKING ENSEMBLE <<", stacking),
        (">> VOTING ENSEMBLE <<", voting)
    ]
    
    # --- BENCHMARK LOOP ---
    best_score = 0
    best_model = None
    best_name = ""
    
    for name, model in all_models:
        try:
            acc, std = robust_evaluate(model, X, y)
            print(f"{name:<30} | {acc:.4f}     | {std:.4f}")
            
            if acc > best_score:
                best_score = acc
                best_model = model
                best_name = name
        except Exception as e:
            print(f"{name:<30} | FAILED       | Error: {str(e)[:20]}...")

    print("-" * 60)
    print(f"Winner: {best_name} with {best_score:.4f}")
    
    # --- SAVE WINNER ---
    if best_model:
        print(f"Training {best_name} on full data and saving...")
        best_model.fit(X, y)
        preds = best_model.predict(X_test).astype(int)
        
        sub = pd.read_csv("data/gender_submission.csv")
        sub['Survived'] = preds
        sub.to_csv("data/submission_ultimate.csv", index=False)
        print("Success: Submission saved to 'data/submission_ultimate.csv'")
