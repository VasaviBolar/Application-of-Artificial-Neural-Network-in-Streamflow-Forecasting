import pandas as pd

# Load normalized dataset
df = pd.read_csv('normalized_dataset.csv')

# Parse datetime with dayfirst=True to match dataset format
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)

# Sort and reset index
df = df.sort_values('datetime').reset_index(drop=True)

# Ensure date continuity by reindexing with full date range
full_date_range = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max())
df = df.set_index('datetime').reindex(full_date_range).rename_axis('datetime').reset_index()

# Check and print missing values
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Impute missing values with forward fill if any
df = df.ffill()

# Remove duplicate rows if any and print count
duplicates = df.duplicated().sum()
print(f'Duplicate rows found: {duplicates}')
df = df.drop_duplicates()

# Since already normalized, apply winsorization to z-score columns to cap outliers
def winsorize_series(series, lower_quantile=0.01, upper_quantile=0.99):
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return series.clip(lower, upper)

# Winsorize only normalized z columns to handle outliers without affecting raw normalized scale
df['PRECTOTCORR_z'] = winsorize_series(df['PRECTOTCORR_z'])
df['discharge_cfs_z'] = winsorize_series(df['discharge_cfs_z'])

# Save cleaned dataset
df.to_csv('cleaned_normalized_dataset.csv', index=False)
print("Cleaned normalized dataset saved as cleaned_normalized_dataset.csv")
