# Data Pipeline Logic

## Step 1: Conversion
* **Script**: `scripts/convert_data.py`
* **Action**: Converts `csv` to `parquet` (Snappy compression).
* **Benefit**: Faster I/O and preserves data types.

## Step 2: Cleaning
* **Script**: `scripts/clean_data.py`
* **Imputation**:
    * `Age`: Filled with Median.
    * `Fare`: Filled with Median.
    * `Embarked`: Filled with Mode.
* **Encoding**:
    * `Sex`: Converted to binary (male=0, female=1).

## Step 3: Feature Engineering
* **Script**: `scripts/process_data.py`
* **Logic**: 
    * **Title Extraction**: Regex pulls titles (Mr, Mrs) from Name.
    * **Family Size**: `SibSp` + `Parch` + 1.
    * **IsAlone**: Boolean flag if Family Size is 1.
