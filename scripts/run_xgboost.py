import pandas as pd
import os
from xgboost import XGBClassifier

# Define file paths
train_path = "data/train.csv"
test_path = "data/test.csv"
output_path = "data/gender_submission_XGBoost.csv"

# Load datasets
try:
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
except FileNotFoundError:
    print(f"Error: Could not find '{train_path}' or '{test_path}'.")
    exit()

# Select features
features = ["Pclass", "Sex", "SibSp", "Parch"]

# Convert categorical data (like 'Sex') into numbers
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
y = train_data["Survived"]

# Train the model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# Predict
predictions = model.predict(X_test)

# Create output file
output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})

# Save to the data folder
output.to_csv(output_path, index=False)
print(f"Success: '{output_path}' created.")
