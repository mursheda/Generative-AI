
import pandas as pd
import os

cleaned_csv_directory_path = '3.Financial_extractedCsv_to_cleanedCsv'
wrangled_csv_directory_path = '4.Financial_clean_to_wrangledCsv'

if not os.path.exists(wrangled_csv_directory_path):
    os.makedirs(wrangled_csv_directory_path)

def wrangle_dataframe(df):
    df_wrangled = df.groupby('Financial Term').head(15)
    return df_wrangled

for csv_file in os.listdir(cleaned_csv_directory_path):
    if csv_file.endswith('.csv'):
        cleaned_csv_file_path = os.path.join(cleaned_csv_directory_path, csv_file)
        wrangled_csv_file_path = os.path.join(wrangled_csv_directory_path, f"_{csv_file}")

        df = pd.read_csv(cleaned_csv_file_path)

        df_wrangled = wrangle_dataframe(df)

        df_wrangled.to_csv(wrangled_csv_file_path, index=False)

        print(f"Data wrangling completed for {csv_file}. Saved to {wrangled_csv_file_path}")

