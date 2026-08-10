import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load your CSV file
df = pd.read_csv("results_combined_ordered.csv")

# Define metrics to plot
metrics = [
    ("JS Divergence (labels)", "Jensen-Shannon Divergence"),
    ("Distance Difference", "Distance Difference"),
    ("Energy Distance", "Energy Distance"),
    ("KL Divergence", "Kullback-Leibler Divergence"),
    ("MMD", "Maximum Mean Discrepancy"),
    ("Mahalanobis Distance", "Mahalanobis Distance")
]

# Prepare the dataframe in long format
melted_df = pd.DataFrame()

for metric, label in metrics:
    temp_df = df[["PC Setting", f"QS {metric}", f"SRS {metric}"]].copy()
    temp_df = temp_df.melt(
        id_vars="PC Setting",
        value_vars=[f"QS {metric}", f"SRS {metric}"],
        var_name="Method",
        value_name="Value"
    )
    temp_df["Metric"] = label
    temp_df["Method"] = temp_df["Method"].apply(lambda x: "PCA-QS" if "QS" in x else "SRS")
    melted_df = pd.concat([melted_df, temp_df], ignore_index=True)

# Create output folder
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Set seaborn style
sns.set(style="whitegrid")

# Generate and save plots
for metric in melted_df["Metric"].unique():
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=melted_df[melted_df["Metric"] == metric],
        x="PC Setting",
        y="Value",
        hue="Method"
    )
    plt.title(f"Comparison of {metric} Across PC Settings")
    plt.xlabel("PC Setting")
    plt.ylabel(metric)
    plt.legend(title="Method")
    plt.tight_layout()
    
    # Save plot
    filename = f"{output_dir}/{metric.replace(' ', '_').replace('-', '')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

print(f"✅ All plots saved to ./{output_dir}/")
