import pandas as pd
import numpy as np

def clean_sales_data(df):
    """
    Performs comprehensive preprocessing, duplicate removal, formatting, 
    outlier handling, and generates a data quality report.
    """
    # 1. Initialize the Data Quality Report
    report = {
        "initial_rows": len(df),
        "duplicates_removed": 0,
        "missing_values_filled": 0,
        "outliers_capped": 0,
        "final_rows": 0
    }

    # 2. Duplicate Record Removal
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    report["duplicates_removed"] = initial_len - len(df)

    # 3. Data Formatting Standardization
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Strip hidden whitespace from all text/string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        # Replace empty strings that might have been hiding as spaces with actual NaNs
        df[col] = df[col].replace('', np.nan)

    # Dynamic Date Detection (Updated from previous version)
    possible_date_cols = ['date', 'sales_date', 'transaction_date', 'order_date']
    for col in possible_date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df.rename(columns={col: 'date'}, inplace=True)
            break

    # 4. Missing Value Detection and Handling
    # Count missing values before handling them
    report["missing_values_filled"] = int(df.isnull().sum().sum())
    
    # Drop rows that are literally 100% blank across all columns
    df.dropna(how='all', inplace=True)
    
    # Fill missing numbers with the median (better than mean because it ignores extreme values)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Fill missing categorical/text data with 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    # 5. Outlier Detection (Capping via IQR Method)
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count how many outliers exist in this column
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        report["outliers_capped"] += int(outliers)

        # Cap the outliers (Winsorization) so we don't lose the row, but the number is normalized
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # 6. Finalize Report
    report["final_rows"] = len(df)
    
    # Return BOTH the dataframe and the report dictionary
    return df, report