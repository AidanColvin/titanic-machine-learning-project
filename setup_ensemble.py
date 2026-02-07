import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content.strip())
    print(f"✅ Created: {path}")

def main():
    print(">>> Setting up Advanced Ensemble Logic...\n")

    # --- 1. Advanced Ensemble Script ---
    ensemble_code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_engineer_data():
    # Load raw data to extract advanced features like Ticket frequency
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    
    # Combine for consistent feature engineering
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], sort=False)
    
    # --- FEATURE ENGINEERING ---
    
    # 1. Title Extraction
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # 2. Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 3. Ticket Frequency (CRITICAL for >83%)
    # Groups people who traveled on the exact same ticket (friends/families)
    df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')
    
    # 4. Drop unused and high-cardinality columns
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    
    # 5. Encoding
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
    
    # Split back
    train_df = df[df['is_train'] == 1].drop(columns=['is_train'])
    test_df = df[df['is_train'] == 0].drop(columns=['is_train', 'Survived'])
    
    # Handle Missing Values (Simple imputation for stability)
    imputer = SimpleImputer(strategy='median')
    train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
    test_df = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)
    
    y = train_df['Survived']
    X = train_df.drop(columns=['Survived'])
    
    return X, y, test_df

if __name__ == "__main__":
    print("\\n--- Training Advanced Voting Ensemble ---")
    try:
        X, y, X_test = load_and_engineer_data()
        
        # --- DEFINE MODELS ---
        
        # 1. Random Forest (Robust Baseline)
        clf1 = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=1)
        
        # 2. HistGradientBoosting (State-of-the-art for tabular data, similar to LightGBM)
        clf2 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, max_depth=4, random_state=1)
        
        # 3. Logistic Regression (Linear Baseline)
        # Needs a pipeline to scale data first
        clf3 = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(max_iter=1000))
        ])
        
        # 4. Support Vector Machine (Probability estimates needed for Soft Voting)
        clf4 = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, kernel='rbf', random_state=1))
        ])

        # --- VOTING CLASSIFIER ---
        # Combines predictions. 'soft' voting averages the probabilities.
        eclf = VotingClassifier(
            estimators=[
                ('rf', clf1), 
                ('gb', clf2), 
                ('lr', clf3),
                ('svm', clf4)
            ],
            voting='soft'
        )
        
        # Cross Validation Score
        scores = cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
        print(f"Ensemble Average Accuracy (CV): {scores.mean():.4f}")
        
        # Train & Predict
        eclf.fit(X, y)
        preds = eclf.predict(X_test).astype(int)
        
        # Save
        sub = pd.read_csv("data/gender_submission.csv")
        sub['Survived'] = preds
        sub.to_csv("data/submission_ensemble.csv", index=False)
        print("Saved: data/submission_ensemble.csv")
        print("Note: Submit this file to Kaggle to see if you broke the 83% barrier.")
        
    except Exception as e:
        print(f"Error: {e}")
"""
    write_file("scripts/train_ensemble.py", ensemble_code)
    
    print("\n>>> Setup Complete. Run the ensemble with: python scripts/train_ensemble.py")

if __name__ == "__main__":
    main()
