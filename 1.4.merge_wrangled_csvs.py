import pandas as pd
import glob

csv_files_pattern = '4.Financial_clean_to_wrangledCsv/*.csv'

csv_files = glob.glob(csv_files_pattern)

dataframes = []

for file in csv_files:
    
    df = pd.read_csv(file, index_col=None, header=0)
    dataframes.append(df)

merged_df = pd.concat(dataframes, axis=0, ignore_index=True)
merged_df.to_csv('1.4.merged_csv_files.csv', index=False)

print("CSV files have been merged and saved as '1.4.merged_csv_files.csv'")
