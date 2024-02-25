
import pandas as pd
import os

# Directory containing your cleaned CSV files
cleaned_csv_directory_path = '3.Financial_extractedCsv_to_cleanedCsv'
# Directory where you want to save the wrangled CSV files
wrangled_csv_directory_path = '4.Financial_clean_to_wrangledCsv'

# Ensure the wrangled CSV output directory exists
if not os.path.exists(wrangled_csv_directory_path):
    os.makedirs(wrangled_csv_directory_path)

# Function to wrangle the DataFrame
def wrangle_dataframe(df):
    # Wrangle the data to keep no more than 10 rows for each 'Financial Term'
    df_wrangled = df.groupby('Financial Term').head(15)
    return df_wrangled

# Iterate over cleaned CSV files, wrangle, and save them
for csv_file in os.listdir(cleaned_csv_directory_path):
    if csv_file.endswith('.csv'):
        cleaned_csv_file_path = os.path.join(cleaned_csv_directory_path, csv_file)
        wrangled_csv_file_path = os.path.join(wrangled_csv_directory_path, f"_{csv_file}")

        # Read the cleaned CSV into a DataFrame
        df = pd.read_csv(cleaned_csv_file_path)

        # Wrangle the DataFrame
        df_wrangled = wrangle_dataframe(df)

        # Save the wrangled DataFrame back to a new CSV file
        df_wrangled.to_csv(wrangled_csv_file_path, index=False)

        print(f"Data wrangling completed for {csv_file}. Saved to {wrangled_csv_file_path}")

