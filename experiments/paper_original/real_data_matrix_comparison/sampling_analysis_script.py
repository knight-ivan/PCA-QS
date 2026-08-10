
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile

def analyze_sampling_data(file_path, output_dir):
    df = pd.read_csv(file_path)

    # Prepare data
    df['PCCount'] = df['PCCount'].fillna('SRS')
    df['SamplingMethod'] = df.apply(lambda x: f"{x['Sampling']}-{x['PCCount']}", axis=1)
    metric_columns = ['EnergyDistance', 'MahalanobisDistance', 'MMD', 'KLDivergence']

    # Create output directories
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Group for LaTeX table
    comparison_grouped = df.groupby(['DeduceRate', 'Sampling', 'PCCount'], dropna=False)[metric_columns].mean().reset_index()
    comparison_grouped['PCCount'] = comparison_grouped['PCCount'].fillna('SRS')

    deduce_rate_tables = {}
    for rate in comparison_grouped['DeduceRate'].unique():
        subset = comparison_grouped[comparison_grouped['DeduceRate'] == rate].copy()
        subset.sort_values(by='PCCount', inplace=True)
        deduce_rate_tables[f"DeduceRate {rate}"] = subset.reset_index(drop=True)

    combined_latex_table = pd.concat(
        [table.assign(DeduceRateLabel=name) for name, table in deduce_rate_tables.items()],
        ignore_index=True
    )
    cols_order = ['DeduceRateLabel', 'DeduceRate', 'Sampling', 'PCCount'] + metric_columns
    combined_latex_table = combined_latex_table[cols_order]
    latex_table = combined_latex_table.to_latex(index=False, longtable=True, caption="Comparison of PCA-QS and SRS Sampling Across Deduce Rates", label="tab:sampling_comparison")

    latex_table_path = os.path.join(output_dir, "Sampling_Comparison_Table.tex")
    with open(latex_table_path, "w") as f:
        f.write(latex_table)

    # Generate and save boxplots
    for metric in metric_columns:
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df, x='DeduceRate', y=metric, hue='SamplingMethod')
        plt.title(f'Comparison of Sampling Methods by {metric}')
        plt.xlabel('Deduce Rate')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.legend(title='Method-PCCount', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = os.path.join(figures_dir, f"{metric}_Comparison_Boxplot.png")
        plt.savefig(plot_path)
        plt.close()

    # Zip figures
    zip_path = os.path.join(output_dir, "Sampling_Comparison_Boxplots.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_name in os.listdir(figures_dir):
            file_path = os.path.join(figures_dir, file_name)
            zipf.write(file_path, arcname=file_name)

    print(f"LaTeX table saved to: {latex_table_path}")
    print(f"Boxplots zipped at: {zip_path}")

# Example usage:
# analyze_sampling_data("path/to/your/data.csv", "path/to/output")
