import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

def load_and_engineer_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    loads raw data
    combines for engineering
    extracts titles and ticket frequency
    returns X, y, test_df
    """
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    
    # Mark sets to split later
    train['is_train'] = 1
    test['is_train'] = 0
    
    # Combine (sort=False prevents warning)
    df = pd.concat([train, test], sort=False).reset_index(drop=True)
    
    # --- Feature Engineering ---
    
    # 1. Title Extraction (Regex for word before dot)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # 2. Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 3. Ticket Frequency (Critical for high accuracy)
    # Counts how many people share the same ticket number
    df['Ticket_Freq'] = df.groupby('Ticket')['Ticket'].transform('count')
    
    # 4. Drop noise
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    
    # 5. One-Hot Encoding
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
    
    # --- Split back ---
    
    # Train set: Keep 'Survived', drop marker
    train_df = df[df['is_train'] == 1].drop(columns=['is_train'])
    
    # Test set: Drop 'Survived' (it is NaN anyway) and marker
    test_df = df[df['is_train'] == 0].drop(columns=['is_train', 'Survived'])
    
    # Separate Target
    y = train_df['Survived']
    X = train_df.drop(columns=['Survived'])
    
    # --- Imputation ---
    # Impute missing values (Age, Fare) with median
    # We must use the same imputer for both to ensure consistency
    cols = X.columns
    imputer = SimpleImputer(strategy='median')
    
    X = pd.DataFrame(imputer.fit_transform(X), columns=cols)
    test_df = pd.DataFrame(imputer.transform(test_df), columns=cols)
    
    return X, y, test_df

if __name__ == "__main__":
    print("\n--- Training Robust Advanced Ensemble ---")
    
    try:
        X, y, X_test = load_and_engineer_data()
        
        # --- Model Definitions ---
        
        # 1. Random Forest (The reliable baseline)
        rf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=1)
        
        # 2. HistGradientBoosting (The modern standard, handles non-linearities well)
        gb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, max_depth=4, random_state=1)
        
        # 3. Logistic Regression (The linear check)
        # Pipeline ensures scaling happens inside the cross-validation folds
        lr = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(max_iter=1000))
        ])
        
        # 4. SVM (The boundary finder)
        svm = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, kernel='rbf', random_state=1))
        ])
        
        # --- Voting Ensemble ---
        # Combines all 4 models. 'soft' voting averages their probability outputs.
        # This is more robust than 'hard' voting (majority wins).
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf), 
                ('gb', gb), 
                ('lr', lr),
                ('svm', svm)
            ],
            voting='soft'
        )
        
        # --- Evaluation ---
        print("Running Cross-Validation (5-Folds)...")
        scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
        print(f"Ensemble Average Accuracy: {scores.mean():.4f}")
        
        # --- Final Training & Submission ---
        print("Training final model on full dataset...")
        ensemble.fit(X, y)
        preds = ensemble.predict(X_test).astype(int)
        
        sub = pd.read_csv("data/gender_submission.csv")
        sub['Survived'] = preds
        sub.to_csv("data/submission_ensemble.csv", index=False)
        print("Success: Saved to data/submission_ensemble.csv")
        
    except Exception as e:
        print(f"Error: {e}")
