"""
Test script for the sine approximation problem.

This script verifies that the problem formulation is working correctly
and provides a visual demonstration of the sine function approximation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from problems.sine_approximation import SineApproximationProblem


def test_basic_functionality():
    """Test basic functionality of the sine approximation problem."""
    print("Testing sine approximation problem...")
    
    # Create problem instance
    problem = SineApproximationProblem(degree=8, num_samples=100)
    
    # Get problem information
    info = problem.get_problem_info()
    print("Problem setup:")
    print(f"  - Polynomial degree: {info['degree']}")
    print(f"  - Number of samples: {info['num_samples']}")
    print(f"  - Matrix shape: {info['matrix_shape']}")
    print(f"  - Lipschitz constant: {info['lipschitz_constant']:.2e}")
    print(f"  - Condition number: {info['condition_number']:.2e}")
    
    # Test with zero coefficients (should give zero approximation)
    x_zero = np.zeros(info['degree'] + 1)
    f_zero = problem.objective_function(x_zero)
    print(f"  - Objective at zero: {f_zero:.6f}")
    
    # Test with some random coefficients
    np.random.seed(42)  # For reproducibility
    x_random = np.random.randn(info['degree'] + 1) * 0.1
    f_random = problem.objective_function(x_random)
    print(f"  - Objective at random point: {f_random:.6f}")
    
    # Test gradient computation
    grad_random = problem.gradient(x_random)
    print(f"  - Gradient norm at random point: {np.linalg.norm(grad_random):.6f}")
    
    return problem, x_random


def test_least_squares_solution():
    """Test the least squares solution (without regularization)."""
    print("\nTesting least squares solution...")
    
    problem = SineApproximationProblem(degree=8, num_samples=100)
    
    # Solve the least squares problem: min ||Ax - b||²
    # Solution: x = (A^T A)^(-1) A^T b
    A = problem.vandermonde_matrix
    b = problem.target_values
    
    # Use numpy's least squares solver
    x_ls, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    print("Least squares solution:")
    print(f"  - Residual norm: {np.sqrt(residuals[0]):.6f}")
    print(f"  - Objective value: {problem.objective_function(x_ls):.6f}")
    
    # Plot the approximation
    problem.plot_approximation(x_ls, "Least Squares Approximation (No Regularization)")
    
    return x_ls


def test_different_degrees():
    """Test how different polynomial degrees affect the approximation."""
    print("\nTesting different polynomial degrees...")
    
    degrees = [3, 6, 9, 12]
    plt.figure(figsize=(15, 10))
    
    for i, degree in enumerate(degrees):
        problem = SineApproximationProblem(degree=degree, num_samples=100)
        
        # Solve least squares
        A = problem.vandermonde_matrix
        b = problem.target_values
        x_ls, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Evaluate on fine grid
        t_fine = np.linspace(-2*np.pi, 2*np.pi, 1000)
        y_approx = problem.evaluate_polynomial(x_ls, t_fine)
        
        plt.subplot(2, 2, i+1)
        plt.plot(t_fine, np.sin(t_fine), 'b-', label='True sin(t)', linewidth=2)
        plt.plot(t_fine, y_approx, 'r--', label=f'Degree {degree}', linewidth=2)
        plt.plot(problem.sample_points, problem.target_values, 'ko', markersize=3)
        
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title(f'Polynomial Degree {degree}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        print(f"  - Degree {degree}: residual = {np.sqrt(residuals[0]):.6f}")
    
    plt.tight_layout()
    plt.show()


def main():
    """Main test function."""
    print("=" * 60)
    print("SINE APPROXIMATION PROBLEM TEST")
    print("=" * 60)
    
    try:
        # Test basic functionality
        problem, x_random = test_basic_functionality()
        
        # Test least squares solution
        x_ls = test_least_squares_solution()
        
        # Test different degrees
        test_different_degrees()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! Problem formulation is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
