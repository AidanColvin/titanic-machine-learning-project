import pandas as pd
import os

def convert_csv_to_parquet():
    # List of files to convert
    files = ['train.csv', 'test.csv', 'gender_submission.csv']
    
    print("Starting conversion...")
    
    for filename in files:
        # Check if file exists to avoid errors
        if os.path.exists(filename):
            try:
                # 1. Read the CSV file
                df = pd.read_csv(filename)
                
                # 2. Define the output filename (e.g., train.parquet)
                output_filename = filename.replace('.csv', '.parquet')
                
                # 3. Save as Parquet
                # engine='pyarrow' is the standard for speed and compatibility
                # compression='snappy' is the default balance between size and speed
                df.to_parquet(output_filename, engine='pyarrow', compression='snappy')
                
                print(f"✅ Success: '{filename}' converted to '{output_filename}'")
                
            except Exception as e:
                print(f"❌ Error converting '{filename}': {e}")
        else:
            print(f"⚠️ Warning: File '{filename}' not found in current directory.")

    print("\nConversion complete.")

if __name__ == "__main__":
    convert_csv_to_parquet()
