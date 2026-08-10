import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'merged_corr_pca_df' is your dataframe with 'method', 'model', and performance metric columns

# Melt the data for plotting
metrics = ['accuracy', 'auc', 'f1', 'tpr', 'fpr', 'fnr']
melted = merged_corr_pca_df.melt(id_vars=['method', 'model'], value_vars=metrics,
                                 var_name='Metric', value_name='Value')

# Define color palette to highlight QS as light blue
palette = {'QS': 'lightblue', 'SRS': 'lightcoral'}  # QS: light blue, SRS: light coral

# Plot
plt.figure(figsize=(18, 10))
sns.boxplot(data=melted, x='Metric', y='Value', hue='method', palette=palette)
plt.title('Performance Comparison: QS (PCA-QS) vs SRS Across Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
