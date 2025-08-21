"""
Task 2 Experiments: Comprehensive testing of FB, PG, and ASM algorithms.

This script runs experiments with different polynomial degrees (up to 15),
starting points, and regularization parameters to compare the three algorithms.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import time

from problems.sine_approximation import SineApproximationProblem
from algorithms.forward_backward import ForwardBackward
from algorithms.projected_gradient import ProjectedGradient
from algorithms.active_set_method import ActiveSetMethod


def run_degree_comparison():
    """Test different polynomial degrees from 3 to 15."""
    print("=" * 80)
    print("TASK 2: POLYNOMIAL DEGREE COMPARISON (n = 3 to 15)")
    print("=" * 80)
    
    degrees = [3, 5, 7, 9, 11, 13, 15]
    lambda_values = [0.1, 1.0]
    
    results = []
    
    for degree in degrees:
        print(f"\nDegree {degree}:")
        print("-" * 40)
        
        problem = SineApproximationProblem(degree=degree, num_samples=100)
        
        # Test PG
        try:
            start_time = time.time()
            pg = ProjectedGradient(problem)
            pg_result = pg.solve(max_iterations=10000, tolerance=1e-6, verbose=False)
            pg_time = time.time() - start_time
            
            x_pg = pg_result['solution']
            obj_pg = pg_result['final_objective']
            iter_pg = pg_result['iterations']
            l1_pg = np.sum(np.abs(x_pg))
            nonzero_pg = np.sum(np.abs(x_pg) > 1e-10)
            
            print(f"  PG:  obj={obj_pg:.6f}, iter={iter_pg:4d}, L1={l1_pg:.4f}, "
                  f"nonzero={nonzero_pg}, time={pg_time:.2f}s")
            
            results.append({
                'algorithm': 'PG', 'degree': degree, 'objective': obj_pg,
                'iterations': iter_pg, 'l1_norm': l1_pg, 'nonzero': nonzero_pg,
                'time': pg_time, 'solution': x_pg
            })
            
        except Exception as e:
            print(f"  PG:  FAILED - {e}")
        
        # Test ASM
        try:
            start_time = time.time()
            asm = ActiveSetMethod(problem)
            asm_result = asm.solve(max_iterations=1000, tolerance=1e-6, verbose=False)
            asm_time = time.time() - start_time
            
            x_asm = asm_result['solution']
            obj_asm = asm_result['final_objective']
            iter_asm = asm_result['iterations']
            l1_asm = np.sum(np.abs(x_asm))
            nonzero_asm = np.sum(np.abs(x_asm) > 1e-10)
            
            print(f"  ASM: obj={obj_asm:.6f}, iter={iter_asm:4d}, L1={l1_asm:.4f}, "
                  f"nonzero={nonzero_asm}, time={asm_time:.2f}s")
            
            results.append({
                'algorithm': 'ASM', 'degree': degree, 'objective': obj_asm,
                'iterations': iter_asm, 'l1_norm': l1_asm, 'nonzero': nonzero_asm,
                'time': asm_time, 'solution': x_asm
            })
            
        except Exception as e:
            print(f"  ASM: FAILED - {e}")
        
        # Test FB with different lambdas
        for lambda_val in lambda_values:
            try:
                start_time = time.time()
                fb = ForwardBackward(problem, lambda_val)
                fb_result = fb.solve(max_iterations=10000, tolerance=1e-6, verbose=False)
                fb_time = time.time() - start_time
                
                x_fb = fb_result['solution']
                obj_fb = fb_result['final_objective']
                iter_fb = fb_result['iterations']
                l1_fb = np.sum(np.abs(x_fb))
                nonzero_fb = np.sum(np.abs(x_fb) > 1e-10)
                
                print(f"  FB(λ={lambda_val}): obj={obj_fb:.6f}, iter={iter_fb:4d}, "
                      f"L1={l1_fb:.4f}, nonzero={nonzero_fb}, time={fb_time:.2f}s")
                
                results.append({
                    'algorithm': f'FB(λ={lambda_val})', 'degree': degree, 'objective': obj_fb,
                    'iterations': iter_fb, 'l1_norm': l1_fb, 'nonzero': nonzero_fb,
                    'time': fb_time, 'solution': x_fb, 'lambda': lambda_val
                })
                
            except Exception as e:
                print(f"  FB(λ={lambda_val}): FAILED - {e}")
    
    return results


def test_lambda_sensitivity():
    """Test FB algorithm with different lambda values."""
    print("\n" + "=" * 80)
    print("LAMBDA SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    degree = 10
    problem = SineApproximationProblem(degree=degree, num_samples=100)
    lambda_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    results = []
    
    print(f"Testing degree {degree} with different λ values:")
    print("-" * 50)
    
    for lambda_val in lambda_values:
        try:
            fb = ForwardBackward(problem, lambda_val)
            result = fb.solve(max_iterations=10000, tolerance=1e-6, verbose=False)
            
            x = result['solution']
            obj = result['final_objective']
            l1_norm = np.sum(np.abs(x))
            nonzero = np.sum(np.abs(x) > 1e-10)
            
            # Compute approximation error
            t_test = np.linspace(-2*np.pi, 2*np.pi, 1000)
            y_true = np.sin(t_test)
            y_approx = problem.evaluate_polynomial(x, t_test)
            rmse = np.sqrt(np.mean((y_true - y_approx)**2))
            
            print(f"λ={lambda_val:5.2f}: obj={obj:.6f}, L1={l1_norm:.4f}, "
                  f"nonzero={nonzero:2d}, RMSE={rmse:.6f}")
            
            results.append({
                'lambda': lambda_val, 'objective': obj, 'l1_norm': l1_norm,
                'nonzero': nonzero, 'rmse': rmse, 'solution': x
            })
            
        except Exception as e:
            print(f"λ={lambda_val:5.2f}: FAILED - {e}")
    
    return results


def create_comparison_plots(degree_results, lambda_results):
    """Create comparison plots."""
    print("\n" + "=" * 80)
    print("CREATING COMPARISON PLOTS")
    print("=" * 80)
    
    # Ensure results directory exists
    os.makedirs('results/plots', exist_ok=True)
    
    # Plot 1: Performance vs Degree
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Group by algorithm
    alg_data = {}
    for result in degree_results:
        alg = result['algorithm']
        if alg not in alg_data:
            alg_data[alg] = {'degrees': [], 'objectives': [], 'iterations': [], 'times': []}
        alg_data[alg]['degrees'].append(result['degree'])
        alg_data[alg]['objectives'].append(result['objective'])
        alg_data[alg]['iterations'].append(result['iterations'])
        alg_data[alg]['times'].append(result['time'])
    
    # Objective vs degree
    ax = axes[0, 0]
    for alg, data in alg_data.items():
        ax.plot(data['degrees'], data['objectives'], 'o-', label=alg, linewidth=2, markersize=6)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Final Objective Value')
    ax.set_title('Objective Value vs Degree')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Iterations vs degree
    ax = axes[0, 1]
    for alg, data in alg_data.items():
        ax.plot(data['degrees'], data['iterations'], 'o-', label=alg, linewidth=2, markersize=6)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Number of Iterations')
    ax.set_title('Iterations vs Degree')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Computation time vs degree
    ax = axes[1, 0]
    for alg, data in alg_data.items():
        ax.plot(data['degrees'], data['times'], 'o-', label=alg, linewidth=2, markersize=6)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('Time vs Degree')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Condition number vs degree
    ax = axes[1, 1]
    degrees = []
    cond_numbers = []
    for deg in sorted(set(r['degree'] for r in degree_results)):
        problem = SineApproximationProblem(degree=deg, num_samples=100)
        degrees.append(deg)
        cond_numbers.append(problem.condition_number)
    
    ax.semilogy(degrees, cond_numbers, 'ro-', linewidth=2, markersize=6)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Condition Number')
    ax.set_title('Problem Condition Number vs Degree')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/task2_degree_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/task2_degree_comparison.png")
    plt.show()
    
    # Plot 2: Lambda sensitivity
    if lambda_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        lambdas = [r['lambda'] for r in lambda_results]
        objectives = [r['objective'] for r in lambda_results]
        l1_norms = [r['l1_norm'] for r in lambda_results]
        nonzeros = [r['nonzero'] for r in lambda_results]
        rmses = [r['rmse'] for r in lambda_results]
        
        axes[0, 0].semilogx(lambdas, objectives, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('λ')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].set_title('Objective vs λ')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].semilogx(lambdas, l1_norms, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('λ')
        axes[0, 1].set_ylabel('L1 Norm')
        axes[0, 1].set_title('Sparsity vs λ')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].semilogx(lambdas, nonzeros, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('λ')
        axes[1, 0].set_ylabel('Non-zero Coefficients')
        axes[1, 0].set_title('Active Coefficients vs λ')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].loglog(lambdas, rmses, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('λ')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('Approximation Quality vs λ')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/task2_lambda_sensitivity.png', dpi=300, bbox_inches='tight')
        print("Saved: results/plots/task2_lambda_sensitivity.png")
        plt.show()


def plot_function_approximations(degree_results):
    """Plot function approximations for degree 10."""
    # Find results for degree 10
    deg_10_results = [r for r in degree_results if r['degree'] == 10]
    
    if not deg_10_results:
        return
    
    problem = SineApproximationProblem(degree=10, num_samples=100)
    t_fine = np.linspace(-2*np.pi, 2*np.pi, 1000)
    y_true = np.sin(t_fine)
    
    n_methods = len(deg_10_results)
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    for i, result in enumerate(deg_10_results):
        method = result['algorithm']
        x_sol = result['solution']
        y_approx = problem.evaluate_polynomial(x_sol, t_fine)
        rmse = np.sqrt(np.mean((y_true - y_approx)**2))
        
        # Top: function approximation
        axes[0, i].plot(t_fine, y_true, 'b-', label='True sin(t)', linewidth=2)
        axes[0, i].plot(t_fine, y_approx, 'r--', label=method, linewidth=2)
        axes[0, i].plot(problem.sample_points, problem.target_values, 'ko', markersize=3)
        axes[0, i].set_xlabel('t')
        axes[0, i].set_ylabel('y')
        axes[0, i].set_title(f'{method}\nRMSE = {rmse:.6f}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Bottom: coefficients
        axes[1, i].bar(range(len(x_sol)), x_sol, alpha=0.7)
        axes[1, i].set_xlabel('Coefficient Index')
        axes[1, i].set_ylabel('Coefficient Value')
        axes[1, i].set_title(f'Coefficients (L1 = {result["l1_norm"]:.4f})')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/task2_approximations.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/task2_approximations.png")
    plt.show()


def main():
    """Main function."""
    print("Starting Task 2 Comprehensive Experiments...")
    
    try:
        # Run experiments
        degree_results = run_degree_comparison()
        lambda_results = test_lambda_sensitivity()
        
        # Create plots
        create_comparison_plots(degree_results, lambda_results)
        plot_function_approximations(degree_results)
        
        print("\n" + "=" * 80)
        print("TASK 2 EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Summary
        print("\nSummary:")
        print(f"- Tested {len(set(r['degree'] for r in degree_results))} different polynomial degrees")
        print(f"- Tested {len(lambda_results)} different λ values")
        print(f"- Compared {len(set(r['algorithm'] for r in degree_results))} algorithms")
        print("- Generated comparison plots in results/plots/")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
