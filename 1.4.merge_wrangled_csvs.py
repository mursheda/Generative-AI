import pandas as pd
import glob

# Define the pattern to match your CSV files, assuming they are in the current directory
csv_files_pattern = '4.Financial_clean_to_wrangledCsv/*.csv'

# Use glob to find all files matching the pattern
csv_files = glob.glob(csv_files_pattern)

# List to hold data from each CSV file
dataframes = []

# Loop over the list of csv files
for file in csv_files:
    
    # Read the CSV file and append it to the list of dataframes
    df = pd.read_csv(file, index_col=None, header=0)
    dataframes.append(df)

# Concatenate all dataframes in the list into one
merged_df = pd.concat(dataframes, axis=0, ignore_index=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('1.4.merged_csv_files.csv', index=False)

print("CSV files have been merged and saved as '1.4.merged_csv_files.csv'")
