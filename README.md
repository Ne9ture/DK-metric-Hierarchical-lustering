# DK-metric-Hierarchical-lustering
Hierarchical clustering algorithms for interval-valued data based on DK metric
# Project Title: Interval-valued Data Clustering

## Overview
This repository contains all the necessary files and data used for clustering interval-valued data using various clustering algorithms based on different kernel metrics. The project explores different methodologies, including case optimizations, Monte Carlo simulations, bootstrap methods, and practical tests on diverse datasets.

## Folder Structure

### Code
This directory contains the source code files for the clustering algorithms. Each file corresponds to different cases (3-5) of optimization processes for the clustering algorithms under five different kernel settings.

- `case_optimizations.py` - Contains the code for optimizing clustering algorithms across different kernel configurations.
- `monte_carlo_simulations.py` - Implements the Monte Carlo simulations for estimating the performance stability of each clustering approach.
- `bootstrap_methods.py` - Provides the implementation of bootstrap techniques for assessing the robustness of clustering results.

### Data
Contains all datasets used in the experiments. This includes real-world data as well as synthetic datasets created for testing purposes.

- `fungi_dataset.csv` - Real-world dataset consisting of interval-valued data collected from fungi studies.
- `stock_dataset.csv` - Stock market data used to demonstrate the application of clustering methods in financial analysis.
- `feature_datasets` - Directory containing various datasets with different features extracted for clustering experiments.

### Plots
This folder stores output images from the clustering processes, showcasing the results and the effectiveness of different kernels.

- `kernel_comparison.png` - Visual comparison of clustering outcomes across different kernels.
- `bootstrap_results.png` - Results from the bootstrap analysis, illustrating the statistical reliability of the clustering.

## Running the Code

To run the clustering algorithms and reproduce the experiments, follow these steps:

1. **Install Dependencies**:
   Ensure that Python and all required libraries are installed, including NumPy, Pandas, Matplotlib, and SciPy.

2. **Execute the Scripts**:
   Run each script separately to perform the clustering operations and generate results. For example, use the following command to run the case optimization process:
   ```bash
   python case_optimizations.py
