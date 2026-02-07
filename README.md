@misc{titanic,
    author = {Aidan Colvin},
    title = {Titanic - Machine Learning from Disaster},
    year = {2026},
    howpublished = {\url{https://kaggle.com/competitions/titanic}},
    note = {Kaggle}
}

# Titanic Machine Learning Project

## Repository Structure
* **data/**: Contains raw CSVs, cleaned Parquet files, and final processed data.
* **scripts/**: Python scripts for the data pipeline.
* **docs/**: Documentation for data and pipeline logic.

## How to Run the Pipeline
1. `python scripts/convert_data.py` (Convert CSV to Parquet)
2. `python scripts/clean_data.py` (Impute missing values)
3. `python scripts/process_data.py` (Feature Engineering: Titles, Family Size)
