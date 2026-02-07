@misc{titanic,
    author = {Aidan Colvin},
    title = {Titanic - Machine Learning from Disaster},
    year = {2026},
    howpublished = {\url{https://kaggle.com/competitions/titanic}},
    note = {Kaggle}
}

# Titanic Machine Learning Project

=======================================================
      TITANIC MODEL BENCHMARK (5-FOLD CV)
=======================================================
Features Used: Ticket_Freq, Title_Group, FamilySize
-------------------------------------------------------
Model Name                | Accuracy   | Std Dev   
-------------------------------------------------------
Random Forest             | 0.8227     | 0.0169
HistGradientBoosting      | 0.8306     | 0.0308
Logistic Regression       | 0.8260     | 0.0280
Support Vector Machine    | 0.8249     | 0.0200
>> VOTING ENSEMBLE <<     | 0.8395     | 0.0184
-------------------------------------------------------
Note: The 'Voting Ensemble' combines the mathematical
strengths of all 4 models to reduce variance/errors.