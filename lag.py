import pandas as pd
from statsmodels.tsa.stattools import acf

# Load normalized dataset
df = pd.read_csv("cleaned_normalized_dataset.csv", parse_dates=["datetime"])
df = df.set_index("datetime")

# ---------------------------
# Parameters
# ---------------------------
manual_lags_discharge = [1, 2, 3]    # Manual lags for discharge_cfs_z
manual_lags_precip = [1, 2, 3, 4, 5, 6, 7]  # Manual lags for PRECTOTCORR_z
auto_threshold_discharge = 0.3       # Autocorrelation threshold for discharge
auto_threshold_precip = 0.1          # Autocorrelation threshold for precipitation
max_lags = 14                        # Maximum lags to check for autocorrelation

# ---------------------------
# Function to suggest lags based on autocorrelation
# ---------------------------
def suggest_lags(series, threshold, max_lags):
    autocorr = acf(series, nlags=max_lags, fft=False)
    return [i for i, val in enumerate(autocorr[1:], start=1) if abs(val) >= threshold]

# ---------------------------
# Automatic lags
# ---------------------------
auto_lags_discharge = suggest_lags(df['discharge_cfs_z'], threshold=auto_threshold_discharge, max_lags=max_lags)
auto_lags_precip = suggest_lags(df['PRECTOTCORR_z'], threshold=auto_threshold_precip, max_lags=max_lags)

# Combine manual + automatic lags (remove duplicates)
final_lags_discharge = sorted(list(set(manual_lags_discharge + auto_lags_discharge)))
final_lags_precip = sorted(list(set(manual_lags_precip + auto_lags_precip)))

print(f"Final lags for discharge_cfs_z: {final_lags_discharge}")
print(f"Final lags for PRECTOTCORR_z: {final_lags_precip}")

# ---------------------------
# Function to create lag features
# ---------------------------
def create_lags(data, column, lag_list):
    for lag in lag_list:
        data[f"{column}_lag{lag}"] = data[column].shift(lag)
    return data

# ---------------------------
# Create lag features
# ---------------------------
df = create_lags(df, "discharge_cfs_z", final_lags_discharge)
df = create_lags(df, "PRECTOTCORR_z", final_lags_precip)

# ---------------------------
# Backward fill missing lag values for first rows
# ---------------------------
df = df.bfill().reset_index()

# ---------------------------
# Save final dataset
# ---------------------------
df.to_csv("dataset_with_combined_lags.csv", index=False)
print("Dataset with combined manual + auto lags saved as dataset_with_combined_lags.csv")
