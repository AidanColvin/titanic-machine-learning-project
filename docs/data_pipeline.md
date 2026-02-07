# Data Pipeline Documentation

## 1. Conversion
Script: `scripts/convert_data.py`
* Inputs: `data/*.csv`
* Outputs: `data/*.parquet`
* Description: Converts text-based CSVs to binary Parquet format for efficiency.

## 2. Cleaning
Script: `scripts/clean_data.py`
* Inputs: `data/*.parquet`
* Outputs: `data/*_cleaned.parquet`
* Operations: 
    * Fill Age (Median)
    * Fill Fare (Median)
    * Fill Embarked (Mode)
    * Encode Sex (0/1)

## 3. Processing
Script: `scripts/process_data.py`
* Inputs: `data/*_cleaned.parquet`
* Outputs: `data/*_processed.parquet`
* Operations:
    * Extract Title from Name
    * Calculate FamilySize
    * Create IsAlone flag
