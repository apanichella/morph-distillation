import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV
from tqdm import tqdm
from scipy.stats import mannwhitneyu
import itertools



# Define the function to compute Vargha-Delaney A measure
def vargha_delaney_a(x, y):
    m, n = len(x), len(y)
    r = np.sum([sum(j < i for j in y) for i in x])
    r /= (m * n)
    return r

# Read the CSV file
df = pd.read_csv("mo_pareto_fronts.csv")

# Normalize the objectives (except 'Algorithm' and 'Seed')
objectives = ['Model Size', 'Accuracy', 'FLOPs', 'Prediction Flips']

# Convert 'Accuracy' to positive values for minimization problems before normalization
df['Accuracy'] = -df['Accuracy']

# Calculate min and max for each objective
min_vals = df[objectives].min()
max_vals = df[objectives].max()

# Normalize the objectives
df[objectives] = (df[objectives] - min_vals) / (max_vals - min_vals)

# Get unique seeds and algorithms
seeds = df['Seed'].unique()
algorithms = df['Algorithm'].unique()

# Determine the worst point (maximum values for each objective) after normalization
worst_point = [1, 1, 1, 1]
print("Worst point for HV calculation (normalized):", worst_point)

# Initialize a dictionary to store HV values
hv_results = {alg: [] for alg in algorithms}

# Calculate HV for each run and algorithm with progress bar
for seed in tqdm(seeds, desc="Seeds"):
    for algorithm in tqdm(algorithms, desc="Algorithms", leave=False):
        subset = df[(df['Seed'] == seed) & (df['Algorithm'] == algorithm)]
        if not subset.empty:
            points = subset[objectives].values
            hv = HV(ref_point=worst_point)(points)
            hv_results[algorithm].append(hv)

# Function to compute IQR
def compute_iqr(data):
    q75, q25 = np.percentile(data, [75, 25])
    return q75 - q25

# Calculate median and IQR for each algorithm
hv_summary = {}
for algorithm, hvs in hv_results.items():
    median_hv = np.median(hvs)
    iqr_hv = compute_iqr(hvs)
    hv_summary[algorithm] = {'Median HV': median_hv, 'IQR HV': iqr_hv}

# Print the results
print(hv_summary)

# Initialize a dictionary to store statistical test results
statistical_tests = {}

# Perform Wilcoxon-Mann-Whitney test and Vargha-Delaney A measure
base_algorithm = 'AGEMOEA'
for algorithm in algorithms:
    if algorithm != base_algorithm:
        u_stat, p_value = mannwhitneyu(hv_results[base_algorithm], hv_results[algorithm], alternative='two-sided')
        vd_a = vargha_delaney_a(hv_results[base_algorithm], hv_results[algorithm])
        statistical_tests[algorithm] = {
            'U Statistic': u_stat,
            'P Value': p_value,
            'Vargha-Delaney A': vd_a
        }

# Print the statistical test results
print(statistical_tests)