# Numerical Optimization Project 3: Lasso Optimization Algorithms

This project implements and compares three optimization algorithms for Lasso-type problems:

- **Forward-Backward (FB)**: Proximal gradient method for penalized Lasso
- **Projected Gradient (PG)**: For constrained Lasso
- **Active-Set Method (ASM)**: Alternative approach for comparison

## Problem Formulation

We approximate the sine function over `[-2π, 2π]` using a polynomial of degree `n` while encouraging sparsity in the coefficient vector.

### Mathematical Formulation

The approximation uses the polynomial:

```
φ(x; t) = Σᵢ₌₀ⁿ xᵢ tⁱ,  x ∈ ℝⁿ⁺¹
```

The objective function is:

```
f(x) = ½ ||Ax - b||²
```

Where:

- `A` is the Vandermonde matrix
- `b` is the target vector `bⱼ = sin(aⱼ)`
- `aⱼ` are uniformly spaced sample points

### Problem Types

1. **Penalized Lasso**: `min f(x) + λ||x||₁`
2. **Constrained Lasso**: `min f(x)` subject to `||x||₁ ≤ 1`

## Project Structure

```
num_opt_proj_3/
├── src/
│   ├── algorithms/     # Optimization algorithms (FB, PG, ASM)
│   ├── problems/       # Problem formulations
│   ├── utils/          # Utility functions
│   └── experiments/    # Experiment scripts
├── results/
│   ├── plots/          # Generated plots
│   └── data/           # Numerical results
├── docs/               # Documentation and reports
├── Makefile            # Build and run commands
└── pyproject.toml      # Project configuration
```

## Quick Start

1. **Setup environment**:

   ```bash
   make setup
   ```

2. **Test basic functionality**:

   ```bash
   make run-basic
   ```

3. **Run all experiments**:

   ```bash
   make run-all
   ```

## Available Commands

- `make help` - Show all available commands
- `make install` - Install dependencies
- `make test` - Run tests
- `make clean` - Clean generated files
- `make run-basic` - Run basic sine function test
- `make run-all` - Run all optimization experiments

## Dependencies

- Python ≥ 3.13
- NumPy ≥ 1.24.0
- Matplotlib ≥ 3.7.0
- SciPy ≥ 1.11.0
- Pandas ≥ 2.0.0

## Development

For development dependencies:

```bash
make dev-install
```

## Tasks

- **Task 1**: Implement FB for penalized Lasso and PG for constrained Lasso
- **Task 2**: Apply FB, PG, and ASM to both problem types
- **Task 3**: Analyze condition number and implement pre-conditioning
- **Task 4**: Run algorithms on pre-conditioned problems

## License

This project is for educational purposes as part of a numerical optimization course.
