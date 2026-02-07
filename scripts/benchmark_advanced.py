import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def impute_age_by_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe with age and title
    return dataframe with age imputed by median of title group
    more accurate than overall median
    """
    df = df.copy()
    # Ensure Title exists
    if 'Title' not in df.columns:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Impute
    for title in df['Title'].unique():
        median_age = df[df['Title'] == title]['Age'].median()
        # Fallback to overall median if title group is empty/NaN
        if pd.isna(median_age):
            median_age = df['Age'].median()
        df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = median_age
    return df

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with advanced engineered features
    ticket freq, fare per person, deck, title group
    """
    df = df.copy()
    
    # 1. Ticket Frequency (Group Size)
    df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')
    
    # 2. Fare Per Person
    df['FarePerPerson'] = df['Fare'] / df['TicketFreq']
    
    # 3. Title Grouping (Map to Ordinal)
    if 'Title' not in df.columns:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
    title_mapping = {
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
        'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1,
        'Countess': 5, 'Ms': 1, 'Lady': 5, 'Jonkheer': 4,
        'Don': 4, 'Dona': 5, 'Mme': 2, 'Capt': 4, 'Sir': 4
    }
    # Map and fill unknown titles with 'Rare' (4)
    df['TitleGroup'] = df['Title'].map(title_mapping).fillna(4)
    
    # 4. Cabin Deck
    df['Deck'] = df['Cabin'].str[0] if 'Cabin' in df.columns else 'U'
    deck_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'U': 8}
    # Check for decks not in mapping (fill U)
    df['Deck'] = df['Deck'].map(deck_mapping).fillna(8)
    
    # 5. Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 6. IsMother
    df['IsMother'] = ((df['Sex'] == 'female') & 
                      (df['Parch'] > 0) & 
                      (df['Age'] > 18)).astype(int)
                      
    return df

def load_and_prep_data():
    """
    load raw data
    apply advanced engineering
    return X, y
    """
    # Load
    try:
        train = pd.read_csv("data/train.csv")
        test = pd.read_csv("data/test.csv")
    except FileNotFoundError:
        print("Error: Files not found in data/ folder")
        exit()

    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], sort=False).reset_index(drop=True)
    
    # --- 1. Smart Imputation ---
    df = impute_age_by_title(df)
    
    # --- 2. Advanced Feature Engineering ---
    df = create_advanced_features(df)
    
    # --- 3. Encoding & Cleaning ---
    # Drop raw columns
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Title', 'AgeBin', 'FareBin'] 
    # Note: We drop AgeBin/FareBin if they exist to rely on tree models handling continuous vars
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Encode Categoricals (Sex, Embarked)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Fill remaining missing (FarePerPerson can be NaN if Fare is NaN)
    cols = df.columns
    imputer = SimpleImputer(strategy='median')
    df = pd.DataFrame(imputer.fit_transform(df), columns=cols)
    
    # Split
    train_df = df[df['is_train'] == 1].drop(columns=['is_train'])
    y = train_df['Survived']
    X = train_df.drop(columns=['Survived'])
    
    return X, y

if __name__ == "__main__":
    X, y = load_and_prep_data()
    
    print("\n" + "="*65)
    print("      TITANIC ADVANCED BENCHMARK (STACKING & XGBOOST)")
    print("="*65)
    print("New Features: TicketFreq, FarePerPerson, TitleGroup, Deck, IsMother")
    print("-" * 65)
    
    # --- DEFINE MODELS ---
    
    # 1. Previous Strong Models
    rf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=1)
    lr = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=1000))])
    svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, kernel='rbf', random_state=1))])
    
    # 2. NEW: XGBoost (Gradient Boosting on steroids)
    xgb = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05, 
        subsample=0.8, colsample_bytree=0.8, random_state=1, eval_metric='logloss'
    )
    
    # 3. NEW: LightGBM (Fast, high accuracy)
    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, random_state=1, verbose=-1
    )
    
    # 4. NEW: Tuned Gradient Boosting
    gb_tuned = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, random_state=1
    )
    
    # 5. NEW: Stacking Ensemble (The "Kaggle Winner" approach)
    # Uses RF, XGB, LGBM, GB as base, LogReg to combine them
    estimators = [
        ('rf', rf),
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('gb', gb_tuned)
    ]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    models = [
        ("Random Forest (Baseline)", rf),
        ("Logistic Regression", lr),
        ("Support Vector Machine", svm),
        ("XGBoost", xgb),
        ("LightGBM", lgbm),
        ("GradientBoosting (Tuned)", gb_tuned),
        (">> STACKING ENSEMBLE <<", stacking)
    ]
    
    # --- RUN BENCHMARK ---
    
    print(f"{'Model Name':<30} | {'Accuracy':<10} | {'Std Dev':<10}")
    print("-" * 65)
    
    for name, model in models:
        # 5-Fold CV
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"{name:<30} | {scores.mean():.4f}     | {scores.std():.4f}")
    
    print("-" * 65)
    print("Tip: If Stacking or XGBoost > 84%, submit that file to Kaggle.")
