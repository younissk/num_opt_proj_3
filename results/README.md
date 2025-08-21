# Results Directory - Numerical Optimization Project 3

This directory contains all experimental results, visualizations, and data for Tasks 1 and 2 of the Lasso optimization algorithms project.

## Directory Structure

```
results/
├── README.md                           # This file
├── data/                               # Experimental data
│   ├── task1_results.pkl              # Task 1 complete results (binary)
│   ├── task1_results.json             # Task 1 summary (human-readable)
│   ├── task2_results.pkl              # Task 2 complete results (binary)
│   └── task2_results.json             # Task 2 summary (human-readable)
└── plots/                              # Generated visualizations
    ├── basic_sine_test.png            # Basic sine function test
    ├── task1_comprehensive.png        # Task 1 comprehensive analysis
    ├── task2_degree_comparison.png    # Task 2 degree comparison
    ├── task2_lambda_sensitivity.png   # Task 2 lambda sensitivity
    ├── task2_performance_analysis.png # Task 2 performance analysis
    ├── task2_lambda_analysis.png      # Task 2 lambda analysis
    └── task2_approximation_comparison.png # Task 2 approximation quality
```

## Generated Images

### Task 1: Forward-Backward and Projected Gradient

- **`task1_comprehensive.png`**: 6-panel comprehensive analysis including:
  - Function approximations comparison
  - Coefficient patterns across algorithms
  - L1 norm vs regularization parameter
  - Non-zero coefficients vs regularization
  - Objective values vs regularization
  - Convergence behavior comparison

### Task 2: Active-Set Method and Comprehensive Analysis

- **`task2_performance_analysis.png`**: 4-panel performance analysis:
  - Objective values vs polynomial degree
  - Iterations vs polynomial degree
  - Computation time vs polynomial degree
  - Condition number vs polynomial degree

- **`task2_lambda_analysis.png`**: 4-panel lambda sensitivity:
  - Objective values vs λ
  - L1 norm vs λ
  - Non-zero coefficients vs λ
  - Approximation quality (RMSE) vs λ

- **`task2_approximation_comparison.png`**: 8-panel approximation quality:
  - Function approximations for degrees 3, 7, 11, 15
  - Coefficient patterns for each degree
  - Comparison across all algorithms (FB, PG, ASM)

### Task 3: Condition Number Analysis and Pre-conditioning
- ### Task 4: Pre-conditioned Algorithm Performance
- **`task4_preconditioned_performance.png`**: Composite performance metrics (iterations, objective, time, residual) across methods and degrees
- **`task4_approximation_quality.png`**: Function approximation quality at degree 12 for best solutions per algorithm and method


- **`task3_condition_analysis.png`**: 4-panel condition number analysis:
  - Condition number vs polynomial degree (3-20)
  - Log-log growth rate analysis
  - Growth rate patterns
  - Singular value distributions for selected degrees

- **`task3_performance_comparison.png`**: 4-panel performance comparison:
  - Objective values comparison (original vs pre-conditioned)
  - Iterations comparison
  - Solution sparsity comparison
  - Condition number comparison

- **`task3_comprehensive_analysis.png`**: 4-panel comprehensive analysis:
  - Condition number comparison across methods
  - Pre-conditioning effectiveness
  - Logarithmic improvement analysis
  - Average effectiveness summary

## Data Files

### Task 1 Results

- **Complete data**: `task1_results.pkl` (2.5KB)
- **Summary**: `task1_results.json` (2.7KB)
- **Contents**: FB results for λ ∈ [0.1, 0.5, 1.0, 2.0, 5.0], PG results, ASM results

### Task 2 Results

- **Complete data**: `task2_results.pkl` (1.6MB)
- **Summary**: `task2_results.json` (5.6KB)
- **Contents**: Results for degrees 3-15, lambda sensitivity analysis, algorithm comparisons

### Task 3 Results

- **Complete data**: `task3_results.pkl` (~2.0MB)
- **Summary**: `task3_results.json` (~8.0KB)
- **Contents**: Condition number analysis for degrees 3-20, pre-conditioning effectiveness comparison, algorithm performance on pre-conditioned problems

### Task 4 Results
- **Complete data**: `task4_results.pkl` (~3.0MB)
- **Summary**: `task4_results.json` (~12.0KB)
- **Contents**: Runs of FB (λ ∈ {0.1, 0.5, 1.0, 2.0}), PG, ASM on original, diagonal, and SVD pre-conditioned problems across degrees {8, 10, 12, 15}; composite metrics and approximation quality plots.

## Key Findings

### Algorithm Performance

1. **Forward-Backward (FB)**: Good for penalized Lasso, λ controls sparsity
2. **Projected Gradient (PG)**: Efficient for constrained Lasso, good convergence
3. **Active-Set Method (ASM)**: Best approximation quality, fewest iterations

### Problem Conditioning

- **Low degrees (3-7)**: Well-conditioned, all algorithms work well
- **Medium degrees (9-11)**: Some conditioning issues, performance degradation
- **High degrees (13-15)**: Severely ill-conditioned, numerical difficulties

### Pre-conditioning Effectiveness

- **Diagonal method**: Consistent 10-15 orders of magnitude improvement
- **SVD method**: Near-perfect conditioning ($\kappa \approx 1$) across all degrees
- **Cholesky method**: Degrades conditioning for high degrees
- **Recommendation**: Diagonal pre-conditioning provides best balance of effectiveness and efficiency

### Regularization Effects

- Higher λ values increase sparsity but reduce approximation quality
- Optimal λ depends on problem size and desired sparsity level
- All algorithms show similar regularization sensitivity patterns

## Reproducibility

All results are timestamped and include:

- Complete solution vectors
- Convergence histories
- Performance metrics
- Problem parameters
- Algorithm configurations

## Usage

### Viewing Results

- **Images**: Open PNG files in any image viewer
- **JSON data**: Human-readable format for quick analysis
- **Pickle data**: Use Python for detailed analysis

### Python Analysis

```python
import pickle
import json

# Load complete results
with open('results/data/task1_results.pkl', 'rb') as f:
    task1_data = pickle.load(f)

# Load summary data
with open('results/data/task2_results.json', 'r') as f:
    task2_summary = json.load(f)
```

## File Sizes and Formats

- **Images**: High-resolution PNG (300 DPI) suitable for publication
- **Data**: Both binary (pickle) and text (JSON) formats
- **Total size**: ~5.2MB for complete dataset (including Task 3)
- **Compression**: PNG images optimized for quality and size

## Generation

All results were generated using:

- **Script**: `src/experiments/generate_all_images.py`
- **Command**: `make generate-images`
- **Dependencies**: NumPy, Matplotlib, SciPy
- **Environment**: Python 3.13 with UV package manager

## Notes

- All experiments use the same random seed for reproducibility
- Convergence tolerances set to 1e-6 for high accuracy
- Maximum iterations: 50,000 for FB/PG, 1,000 for ASM
- Sample points: 100 uniformly spaced over [-2π, 2π]
