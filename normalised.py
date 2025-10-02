import pandas as pd
from scipy.stats import zscore

# Load the dataset
file_path = "merged_dataset.csv"   # change path if needed
df = pd.read_csv(file_path)

# Apply z-score standardization
df['PRECTOTCORR_z'] = zscore(df['PRECTOTCORR'])
df['discharge_cfs_z'] = zscore(df['discharge_cfs'])

# Save the dataset with normalized values
output_path = "normalized_dataset.csv"
df.to_csv(output_path, index=False)

print(f"Normalized dataset saved to {output_path}")