import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_engineer_data():
    """
    Loads data and applies 'Robust' fixes:
    1. Ticket Frequency (Groups families/friends)
    2. Title Extraction (Captures social status)
    3. Family Size (Captures support network)
    """
    # Load Data
    try:
        train = pd.read_csv("data/train.csv")
        test = pd.read_csv("data/test.csv")
    except FileNotFoundError:
        print("Error: data/train.csv not found. Are you in the right directory?")
        exit()

    # Combine for consistent engineering
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], sort=False).reset_index(drop=True)
    
    # --- FIXES & IMPROVEMENTS ---
    
    # 1. Title Extraction
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    rare = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # 2. Ticket Frequency (The "Secret Sauce" Fix)
    # Counts how many passengers share the same ticket (groups families better than Surname)
    df['Ticket_Freq'] = df.groupby('Ticket')['Ticket'].transform('count')
    
    # 3. Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Drop raw text columns
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
    
    # Split back to train/test
    train_df = df[df['is_train'] == 1].drop(columns=['is_train'])
    y = train_df['Survived']
    X = train_df.drop(columns=['Survived'])
    
    # Impute Missing Values (Robust Median Fill)
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X, y

if __name__ == "__main__":
    X, y = load_and_engineer_data()
    
    print("\n" + "="*55)
    print("      TITANIC MODEL BENCHMARK (5-FOLD CV)")
    print("="*55)
    print("Features Used: Ticket_Freq, Title_Group, FamilySize")
    print("-" * 55)
    
    # --- DEFINE MODELS ---
    
    # 1. Random Forest (Baseline)
    rf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=1)
    
    # 2. HistGradientBoosting (LightGBM equivalent - High Accuracy)
    gb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, max_depth=4, random_state=1)
    
    # 3. Logistic Regression (Linear Baseline - needs scaling)
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000))
    ])
    
    # 4. SVM (Boundary Finder - needs scaling)
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, kernel='rbf', random_state=1))
    ])
    
    # 5. VOTING ENSEMBLE (The "Module" combining best elements)
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr), ('svm', svm)],
        voting='soft'
    )
    
    models = [
        ("Random Forest", rf),
        ("HistGradientBoosting", gb),
        ("Logistic Regression", lr),
        ("Support Vector Machine", svm),
        (">> VOTING ENSEMBLE <<", ensemble)
    ]
    
    # --- RUN BENCHMARK ---
    
    print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Std Dev':<10}")
    print("-" * 55)
    
    for name, model in models:
        # Cross Validation: Runs 5 separate tests on different data splits
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"{name:<25} | {scores.mean():.4f}     | {scores.std():.4f}")
    
    print("-" * 55)
    print("Note: The 'Voting Ensemble' combines the mathematical")
    print("strengths of all 4 models to reduce variance/errors.")
