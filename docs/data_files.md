# Titanic Data Files Documentation

## 1. train.csv
* **Description**: The training dataset containing features and the target variable (`Survived`).
* **Rows**: 891
* **Key Columns**: `Survived` (Target), `Pclass`, `Sex`, `Age`, `Fare`.

## 2. test.csv
* **Description**: The dataset used for model evaluation. Does NOT contain the `Survived` column.
* **Rows**: 418
* **Usage**: Feed into the model to generate predictions.

## 3. gender_submission.csv
* **Description**: A template file provided by Kaggle.
* **Usage**: Example of how to format the final submission.
* **Note**: Do NOT merge this with training data. It is for output formatting only.
