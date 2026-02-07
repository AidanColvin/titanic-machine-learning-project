# Titanic Machine Learning Pipeline

## Folder Structure
* **data/**: Stores all raw CSVs, intermediate Parquet files, and final processed data.
* **scripts/**: Contains Python scripts for the data pipeline.
* **docs/**: Project documentation.

## Pipeline Usage
1. `python scripts/convert_data.py`: Raw CSV -> Parquet.
2. `python scripts/clean_data.py`: Fills missing values, encodes gender.
3. `python scripts/process_data.py`: Feature engineering (Titles, FamilySize).
