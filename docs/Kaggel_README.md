# Titanic - Machine Learning from Disaster

Titanic - Machine Learning from Disaster
[https://www.kaggle.com/c/titanic/overview](https://www.kaggle.com/c/titanic/overview)

---

## The Challenge

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question:

**“what sorts of people were more likely to survive?”**

using passenger data (ie name, age, gender, socio-economic class, etc).

---

## What Data Will I Use in This Competition?

In this competition, you’ll gain access to two similar datasets that include passenger information like name, age, gender, socio-economic class, etc.

One dataset is titled **train.csv** and the other is titled **test.csv**.

### Training Set

Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they survived or not, also known as the “ground truth”.

Your model will be trained using various features, such as:

* Passenger gender
* Passenger class

You may also apply feature engineering to create additional features that could improve model performance.

### Test Set

The test.csv dataset contains similar information but does not disclose the “ground truth” for each passenger.

It’s your job to predict these outcomes.

Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.

Check out the “Data” tab to explore the datasets even further. Once you feel you’ve created a competitive model, submit it to Kaggle to see where your model stands on our leaderboard against other Kagglers.

---

## Dataset Description

### Overview

The dataset has been split into two groups:

* **Training set** (`train.csv`)
* **Test set** (`test.csv`)

### Example Submission

The file `gender_submission.csv` is included as an example of what a valid submission file should look like.

It contains predictions based on the assumption that **all and only female passengers survived**.

---

## Data Dictionary

| Variable | Definition                                 | Key                                            |
| -------- | ------------------------------------------ | ---------------------------------------------- |
| survival | Survival outcome                           | 0 = No, 1 = Yes                                |
| pclass   | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| sex      | Sex                                        |                                                |
| age      | Age in years                               |                                                |
| sibsp    | # of siblings / spouses aboard the Titanic |                                                |
| parch    | # of parents / children aboard the Titanic |                                                |
| ticket   | Ticket number                              |                                                |
| fare     | Passenger fare                             |                                                |
| cabin    | Cabin number                               |                                                |
| embarked | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |

---

## Variable Notes

### pclass

A proxy for socio-economic status (SES):

* 1st = Upper class
* 2nd = Middle class
* 3rd = Lower class

### age

* Age is fractional if less than 1 year old
* If the age was estimated, it appears in the form xx.5

### sibsp

Defines family relationships as follows:

* Sibling: brother, sister, stepbrother, stepsister
* Spouse: husband, wife

Mistresses and fiancés were ignored.

### parch

Defines family relationships as follows:

* Parent: mother, father
* Child: daughter, son, stepdaughter, stepson

Some children traveled only with a nanny; therefore, parch = 0 for them.

---

## Evaluation

### Goal

It is your job to predict if a passenger survived the sinking of the Titanic or not.

For each in the test set, you must predict a 0 or 1 value for the variable.

### Metric

Your score is the percentage of passengers you correctly predict. This is known as accuracy.

### Submission File Format

You should submit a csv file with exactly 418 entries plus a header row. Your submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.

The file should have exactly 2 columns:

* PassengerId (sorted in any order)
* Survived (contains your binary predictions: 1 for survived, 0 for deceased)

```
PassengerId,Survived
892,0
893,1
894,0
Etc.
```
