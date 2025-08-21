"""
Test script for Task 1 algorithms: Forward-Backward and Projected Gradient.

This script tests both optimization algorithms on the sine approximation problem
and compares their performance and solutions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from problems.sine_approximation import SineApproximationProblem
from algorithms.forward_backward import ForwardBackward
from algorithms.projected_gradient import ProjectedGradient


def test_forward_backward():
    """Test the Forward-Backward algorithm for penalized Lasso."""
    print("=" * 60)
    print("TESTING FORWARD-BACKWARD ALGORITHM (Penalized Lasso)")
    print("=" * 60)
    
    # Create problem instance
    problem = SineApproximationProblem(degree=8, num_samples=100)
    
    # Test different lambda values
    lambda_values = [0.1, 0.5, 1.0, 2.0]
    results = {}
    
    for lambda_val in lambda_values:
        print(f"\nTesting λ = {lambda_val}")
        print("-" * 40)
        
        # Create and run algorithm
        fb = ForwardBackward(problem, lambda_val)
        result = fb.solve(max_iterations=50000, tolerance=1e-6, verbose=False)
        
        # Check optimality conditions
        opt_conditions = fb.get_optimality_conditions(result['solution'])
        
        print(f"  Objective value: {result['final_objective']:.6f}")
        print(f"  Proximal residual: {result['final_proximal_residual']:.2e}")
        print(f"  L1 norm: {np.sum(np.abs(result['solution'])):.6f}")
        print(f"  Non-zero coefficients: {np.sum(np.abs(result['solution']) > 1e-10)}")
        print(f"  Subgradient violation: {opt_conditions['subgradient_violation']:.2e}")
        
        results[lambda_val] = result
    
    return results


def test_projected_gradient():
    """Test the Projected Gradient algorithm for constrained Lasso."""
    print("\n" + "=" * 60)
    print("TESTING PROJECTED GRADIENT ALGORITHM (Constrained Lasso)")
    print("=" * 60)
    
    # Create problem instance
    problem = SineApproximationProblem(degree=8, num_samples=100)
    
    # Create and run algorithm
    pg = ProjectedGradient(problem)
    result = pg.solve(max_iterations=50000, tolerance=1e-6, verbose=False)
    
    # Check optimality conditions
    opt_conditions = pg.get_optimality_conditions(result['solution'])
    
    print(f"Objective value: {result['final_objective']:.6f}")
    print(f"Projected gradient: {result['final_projected_gradient']:.2e}")
    print(f"L1 norm: {np.sum(np.abs(result['solution'])):.6f}")
    print(f"Non-zero coefficients: {np.sum(np.abs(result['solution']) > 1e-10)}")
    print(f"Constraint violation: {opt_conditions['constraint_violation']:.2e}")
    print(f"Boundary optimality: {opt_conditions['boundary_optimality']:.2e}")
    
    return result


def compare_solutions():
    """Compare solutions from different approaches."""
    print("\n" + "=" * 60)
    print("COMPARING SOLUTIONS")
    print("=" * 60)
    
    problem = SineApproximationProblem(degree=8, num_samples=100)
    
    # 1. Least squares (no regularization)
    A = problem.vandermonde_matrix
    b = problem.target_values
    x_ls, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # 2. Forward-Backward with λ = 1.0
    fb = ForwardBackward(problem, 1.0)
    result_fb = fb.solve(max_iterations=50000, tolerance=1e-6, verbose=False)
    x_fb = result_fb['solution']
    
    # 3. Projected Gradient
    pg = ProjectedGradient(problem)
    result_pg = pg.solve(max_iterations=50000, tolerance=1e-6, verbose=False)
    x_pg = result_pg['solution']
    
    # Compare properties
    solutions = {
        'Least Squares': x_ls,
        'Forward-Backward (λ=1.0)': x_fb,
        'Projected Gradient': x_pg
    }
    
    print(f"{'Method':<25} {'Objective':<12} {'L1 Norm':<10} {'Non-zero':<10}")
    print("-" * 60)
    
    for name, x in solutions.items():
        obj = problem.objective_function(x)
        l1_norm = np.sum(np.abs(x))
        non_zero = np.sum(np.abs(x) > 1e-10)
        print(f"{name:<25} {obj:<12.6f} {l1_norm:<10.6f} {non_zero:<10}")
    
    # Plot all solutions
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Function approximations
    plt.subplot(1, 3, 1)
    t_fine = np.linspace(-2*np.pi, 2*np.pi, 1000)
    y_true = np.sin(t_fine)
    
    for name, x in solutions.items():
        y_approx = problem.evaluate_polynomial(x, t_fine)
        plt.plot(t_fine, y_approx, '--', label=name, linewidth=2)
    
    plt.plot(t_fine, y_true, 'k-', label='True sin(t)', linewidth=2)
    plt.plot(problem.sample_points, problem.target_values, 'ko', markersize=3)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Function Approximations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Coefficient comparison
    plt.subplot(1, 3, 2)
    x = np.arange(len(x_ls))
    width = 0.25
    
    plt.bar(x - width, x_ls, width, label='Least Squares', alpha=0.8)
    plt.bar(x, x_fb, width, label='Forward-Backward', alpha=0.8)
    plt.bar(x + width, x_pg, width, label='Projected Gradient', alpha=0.8)
    
    plt.xlabel('Coefficient index')
    plt.ylabel('Coefficient value')
    plt.title('Coefficient Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Convergence history
    plt.subplot(1, 3, 3)
    fb_history = result_fb['convergence_history']
    pg_history = result_pg['convergence_history']
    
    plt.semilogy(fb_history['iterations'], fb_history['proximal_residuals'], 
                 'b-', label='FB: Proximal Residual', linewidth=2)
    plt.semilogy(pg_history['iterations'], pg_history['projected_gradients'], 
                 'r-', label='PG: Projected Gradient', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title('Convergence History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main test function."""
    try:
        # Test Forward-Backward algorithm
        fb_results = test_forward_backward()
        
        # Test Projected Gradient algorithm
        pg_result = test_projected_gradient()
        
        # Compare solutions
        compare_solutions()
        
        print("\n" + "=" * 60)
        print("ALL TASK 1 TESTS PASSED!")
        print("Both Forward-Backward and Projected Gradient algorithms are working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
