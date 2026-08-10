import pandas as pd

# Load the original output for reordering
file_path = "results_combined.csv"
df = pd.read_csv(file_path)

# Define new column order: group same-metric types side-by-side
ordered_columns = [
    'PC Setting', 'Run',

    # Label distribution comparison
    'QS JS Divergence (labels)', 'SRS JS Divergence (labels)',

    # Structure preservation
    'QS Distance Difference', 'SRS Distance Difference',
    'QS Energy Distance', 'SRS Energy Distance',
    'QS Normalized Energy Distance', 'SRS Normalized Energy Distance',
    'QS KL Divergence', 'SRS KL Divergence',
    'QS MMD', 'SRS MMD',
    'QS Mahalanobis Distance', 'SRS Mahalanobis Distance',

    # PCA reconstruction
    'PCA Reconstruction Error'
]

# Reorder and save to new CSV
df_ordered = df[ordered_columns]
output_path = "results_combined_ordered.csv"
df_ordered.to_csv(output_path, index=False)

output_path
