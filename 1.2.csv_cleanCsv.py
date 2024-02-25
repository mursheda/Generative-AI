import pandas as pd
import re
import os

# Load the CSV file
extracted_csv_file_path = '2.Financial_pdf_to_csv'  
cleaned_csv_file_path = '3.Financial_extractedCsv_to_cleanedCsv' 

# Ensure the cleaned CSV output directory exists
if not os.path.exists(cleaned_csv_file_path):
    os.makedirs(cleaned_csv_file_path)

# Function to clean a DataFrame
def clean_dataframe(df):
    # Remove rows with any missing values
    df_cleaned = df.dropna()

    # Deduplicate entries, keeping the first occurrence
    df_cleaned = df_cleaned.drop_duplicates(subset=['Financial Text'], keep='first')

    # Text Cleaning: Removing special characters, extra spaces, etc.
    df_cleaned['Financial Text'] = df_cleaned['Financial Text'].str.replace(r"[^a-zA-Z0-9\s-]+", '', regex=True).str.strip()

    # Convert 'Financial Term' to uppercase
    df_cleaned['Financial Term'] = df_cleaned['Financial Term'].str.upper()

    # Remove rows where 'Financial Text' starts with a number
    df_cleaned = df_cleaned[~df_cleaned['Financial Text'].str.match(r'^\d')]

    # Replace newline characters with a space
    df_cleaned['Financial Text'] = df_cleaned['Financial Text'].str.replace('\n', ' ', regex=False)

    # Remove rows with 'Financial Text' having less than 50 characters
    df_cleaned = df_cleaned[df_cleaned['Financial Text'].str.len() > 50]

    # Remove rows where 'Financial Text' starts with specified patterns
    unnecessary_patterns = ['^CSR', '^Gerhard', '^Brauckmann', '^Christian', '^C19bi', '^POCI','BVR', 'MREL']
    pattern_regex = '|'.join(unnecessary_patterns)  # Creates a combined regex pattern
    df_cleaned = df_cleaned[~df_cleaned['Financial Text'].str.match(pattern_regex)]

    return df_cleaned


# Iterate over CSV files in the specified directory, clean, and save them
for csv_file in os.listdir(extracted_csv_file_path):
    if csv_file.endswith('.csv'):
        csv_file_path = os.path.join(extracted_csv_file_path, csv_file)
        cleaned_csv_file = os.path.join(cleaned_csv_file_path, f"{csv_file}")

        # Read the CSV into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Initial data inspection
        print(f"Processing: {csv_file}")
        print("Initial shape:", df.shape)
        print("Initial preview:")
        print(df.head())

        # Clean the DataFrame
        df_cleaned = clean_dataframe(df)

        # Final data inspection
        print("Final shape:", df_cleaned.shape)
        print("Final preview:")
        print(df_cleaned.head())

        # Save the cleaned DataFrame back to a new CSV file
        df_cleaned.to_csv(cleaned_csv_file, index=False)

        print(f"Cleaned data saved to {cleaned_csv_file}\n")
# SH   SH   SH   SH   SH, || ||| || |, - -, - - -,(inside sentence)
# DekaBank, Depositary acts as a one-stop shop for its customers, Notes    Notes,
# Earnings Report     as of September,Post-tax Return, This was driven primarily by higher noninterest expenses (starts with)
# Annual Financial Statements Notes Deutsche Pfandbriefbank AG , Annual Financial  Statements  Annual Financial Statements 1
# Annual Financial Report 2022, (-), KfW HGB 2021 Notes, KGaA Frankfurt am Main, di  United  Services, 
# KfW HGB 2021 Notes  Independent auditors report, social security payments, Dec 2021 EUR, OPPORTUNITIES AND FORECAST REPORT
# Evers  Jnichen,  Book value, Page 3 of, LBB, Disclosure report 2022      