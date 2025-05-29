import numpy as np
import pandas as pd
import torch

# Extract relevant test dataframes from dfs dictionary
df_phone = dfs_test['test_phone_landmarks_all_batch_']
df_tablet = dfs_test['test_tablet_landmarks_all_batch_']

# Concatenate the phone and tablet dataframes
df_combined = pd.concat([df_phone, df_tablet], ignore_index=True)

# Extract gaze vectors (last two columns)
gaze_vectors_phone = df_phone.iloc[:, -2:].values
gaze_vectors_tablet = df_tablet.iloc[:, -2:].values
gaze_vectors_combined = df_combined.iloc[:, -2:].values

# Calculate average magnitudes
avg_magnitude_phone = np.mean(np.linalg.norm(gaze_vectors_phone, axis=1))
avg_magnitude_tablet = np.mean(np.linalg.norm(gaze_vectors_tablet, axis=1))
avg_magnitude_combined = np.mean(np.linalg.norm(gaze_vectors_combined, axis=1))

#obtained from TGGNet_Gazecapture
phone_error = 0.2627 
tablet_error = 0.3038
combined_error = 0.2709


phone_error_cm = phone_error * avg_magnitude_phone
tablet_error_cm = tablet_error * avg_magnitude_tablet
combined_error_cm = combined_error * avg_magnitude_combined
