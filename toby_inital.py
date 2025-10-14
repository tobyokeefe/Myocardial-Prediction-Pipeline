# Initial assessment 
import pandas as pd

demographic_data = pd.read_csv("data/training_data/mixed_demographics.csv")


print("\n--- BASIC INFO ---")
print(f"Shape (rows, columns): {demographic_data.shape}")
print("\nColumn names:")
print(demographic_data.columns.tolist())

print("\nData types and non-null counts:")
demographic_data.info()


print("\n--- SAMPLE ROWS ---")
print(demographic_data.head(10))


print("\n--- SUMMARY STATISTICS ---")
print(demographic_data.describe(include='all'))


print("\n--- MISSING VALUES PER COLUMN ---")
print(demographic_data.isna().sum())


duplicates = demographic_data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")


print("\n--- UNIQUE VALUES PER COLUMN ---")
for col in demographic_data.columns:
    unique_vals = demographic_data[col].nunique()
    print(f"{col}: {unique_vals} unique values")


cat_cols = demographic_data.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    print("\n--- CATEGORY DISTRIBUTIONS ---")
    for col in cat_cols:
        print(f"\n{col} value counts:")
        print(demographic_data[col].value_counts().head(10))