"""
Generate all images for Tasks 1 and 2 and save data.

This script creates comprehensive visualizations for both tasks
and saves all experimental data for future analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime

from problems.sine_approximation import SineApproximationProblem
from algorithms.forward_backward import ForwardBackward
from algorithms.projected_gradient import ProjectedGradient
from algorithms.active_set_method import ActiveSetMethod


def generate_task1_images():
    """Generate comprehensive images for Task 1."""
    print("Generating Task 1 images...")
    
    # Create problem instance
    problem = SineApproximationProblem(degree=8, num_samples=100)
    
    # Test different lambda values for FB
    lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    fb_results = {}
    
    for lambda_val in lambda_values:
        fb = ForwardBackward(problem, lambda_val)
        result = fb.solve(max_iterations=50000, tolerance=1e-6, verbose=False)
        fb_results[lambda_val] = result
    
    # Test PG
    pg = ProjectedGradient(problem)
    pg_result = pg.solve(max_iterations=50000, tolerance=1e-6, verbose=False)
    
    # Test ASM
    asm = ActiveSetMethod(problem)
    asm_result = asm.solve(max_iterations=1000, tolerance=1e-6, verbose=False)
    
    # Plot 1: Function approximations comparison
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: All approximations
    plt.subplot(2, 3, 1)
    t_fine = np.linspace(-2*np.pi, 2*np.pi, 1000)
    y_true = np.sin(t_fine)
    plt.plot(t_fine, y_true, 'k-', label='True sin(t)', linewidth=3)
    
    # Plot FB solutions
    for lambda_val, result in fb_results.items():
        y_approx = problem.evaluate_polynomial(result['solution'], t_fine)
        plt.plot(t_fine, y_approx, '--', label=f'FB(λ={lambda_val})', linewidth=2)
    
    # Plot PG and ASM
    y_pg = problem.evaluate_polynomial(pg_result['solution'], t_fine)
    y_asm = problem.evaluate_polynomial(asm_result['solution'], t_fine)
    plt.plot(t_fine, y_pg, 'r-', label='PG', linewidth=2)
    plt.plot(t_fine, y_asm, 'g-', label='ASM', linewidth=2)
    
    plt.plot(problem.sample_points, problem.target_values, 'ko', markersize=4)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Function Approximations Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Coefficients comparison
    plt.subplot(2, 3, 2)
    x = np.arange(len(pg_result['solution']))
    width = 0.15
    
    # Plot coefficients for different methods
    for i, (lambda_val, result) in enumerate(fb_results.items()):
        plt.bar(x + i*width, result['solution'], width, 
                label=f'FB(λ={lambda_val})', alpha=0.7)
    
    plt.bar(x + len(fb_results)*width, pg_result['solution'], width, 
            label='PG', alpha=0.7)
    plt.bar(x + (len(fb_results)+1)*width, asm_result['solution'], width, 
            label='ASM', alpha=0.7)
    
    plt.xlabel('Coefficient Index')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficient Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: L1 norm vs lambda
    plt.subplot(2, 3, 3)
    lambdas = list(fb_results.keys())
    l1_norms = [np.sum(np.abs(result['solution'])) for result in fb_results.values()]
    plt.semilogx(lambdas, l1_norms, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Regularization Parameter λ')
    plt.ylabel('L1 Norm of Solution')
    plt.title('Sparsity vs Regularization')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Non-zero coefficients vs lambda
    plt.subplot(2, 3, 4)
    non_zeros = [np.sum(np.abs(result['solution']) > 1e-10) for result in fb_results.values()]
    plt.semilogx(lambdas, non_zeros, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Regularization Parameter λ')
    plt.ylabel('Number of Non-zero Coefficients')
    plt.title('Active Coefficients vs Regularization')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Objective values comparison
    plt.subplot(2, 3, 5)
    objectives = [result['final_objective'] for result in fb_results.values()]
    plt.semilogx(lambdas, objectives, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Regularization Parameter λ')
    plt.ylabel('Final Objective Value')
    plt.title('Objective Value vs Regularization')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Convergence comparison
    plt.subplot(2, 3, 6)
    # Plot convergence histories
    for lambda_val, result in fb_results.items():
        history = result['convergence_history']
        plt.semilogy(history['iterations'], history['proximal_residuals'], 
                     'o-', label=f'FB(λ={lambda_val})', linewidth=2)
    
    # Add PG and ASM convergence
    pg_history = pg_result['convergence_history']
    asm_history = asm_result['convergence_history']
    
    plt.semilogy(pg_history['iterations'], pg_history['projected_gradients'], 
                 'r-', label='PG', linewidth=2)
    plt.semilogy(asm_history['iterations'], asm_history['projected_gradients'], 
                 'g-', label='ASM', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Optimality Measure')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/task1_comprehensive.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/task1_comprehensive.png")
    plt.show()
    
    # Save Task 1 data
    task1_data = {
        'fb_results': fb_results,
        'pg_result': pg_result,
        'asm_result': asm_result,
        'problem_info': problem.get_problem_info(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('results/data/task1_results.pkl', 'wb') as f:
        pickle.dump(task1_data, f)
    
    with open('results/data/task1_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in task1_data.items():
            if key == 'fb_results':
                json_data[key] = {}
                for lambda_val, result in value.items():
                    json_data[key][str(lambda_val)] = {
                        'final_objective': float(result['final_objective']),
                        'final_proximal_residual': float(result['final_proximal_residual']),
                        'iterations': int(result['iterations']),
                        'solution': result['solution'].tolist(),
                        'l1_norm': float(np.sum(np.abs(result['solution']))),
                        'non_zero_coeffs': int(np.sum(np.abs(result['solution']) > 1e-10))
                    }
            elif key == 'pg_result':
                json_data[key] = {
                    'final_objective': float(value['final_objective']),
                    'final_projected_gradient': float(value['final_projected_gradient']),
                    'iterations': int(value['iterations']),
                    'solution': value['solution'].tolist(),
                    'l1_norm': float(np.sum(np.abs(value['solution']))),
                    'non_zero_coeffs': int(np.sum(np.abs(value['solution']) > 1e-10))
                }
            elif key == 'asm_result':
                json_data[key] = {
                    'final_objective': float(value['final_objective']),
                    'final_projected_gradient': float(value['final_projected_gradient']),
                    'iterations': int(value['iterations']),
                    'solution': value['solution'].tolist(),
                    'l1_norm': float(np.sum(np.abs(value['solution']))),
                    'non_zero_coeffs': int(np.sum(np.abs(value['solution']) > 1e-10))
                }
            else:
                json_data[key] = value
        
        json.dump(json_data, f, indent=2)
    
    print("Saved Task 1 data to results/data/")
    
    return task1_data


def generate_task2_images():
    """Generate comprehensive images for Task 2."""
    print("Generating Task 2 images...")
    
    # Test different polynomial degrees
    degrees = [3, 5, 7, 9, 11, 13, 15]
    lambda_values = [0.1, 1.0]
    
    all_results = {}
    
    for degree in degrees:
        print(f"Processing degree {degree}...")
        problem = SineApproximationProblem(degree=degree, num_samples=100)
        
        degree_results = {}
        
        # Test PG
        pg = ProjectedGradient(problem)
        pg_result = pg.solve(max_iterations=10000, tolerance=1e-6, verbose=False)
        degree_results['PG'] = pg_result
        
        # Test ASM
        asm = ActiveSetMethod(problem)
        asm_result = asm.solve(max_iterations=1000, tolerance=1e-6, verbose=False)
        degree_results['ASM'] = asm_result
        
        # Test FB with different lambdas
        for lambda_val in lambda_values:
            fb = ForwardBackward(problem, lambda_val)
            fb_result = fb.solve(max_iterations=10000, tolerance=1e-6, verbose=False)
            degree_results[f'FB(λ={lambda_val})'] = fb_result
        
        all_results[degree] = degree_results
    
    # Plot 1: Performance vs Degree (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Organize data by algorithm
    alg_data = {}
    for degree, results in all_results.items():
        for alg_name, result in results.items():
            if alg_name not in alg_data:
                alg_data[alg_name] = {'degrees': [], 'objectives': [], 'iterations': [], 'times': []}
            
            alg_data[alg_name]['degrees'].append(degree)
            alg_data[alg_name]['objectives'].append(result['final_objective'])
            alg_data[alg_name]['iterations'].append(result['iterations'])
            # Estimate time (rough approximation)
            alg_data[alg_name]['times'].append(result['iterations'] * 0.001)  # Rough estimate
    
    # Objective vs degree
    ax = axes[0, 0]
    for alg_name, data in alg_data.items():
        ax.plot(data['degrees'], data['objectives'], 'o-', label=alg_name, linewidth=2, markersize=6)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Final Objective Value')
    ax.set_title('Objective Value vs Degree')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Iterations vs degree
    ax = axes[0, 1]
    for alg_name, data in alg_data.items():
        ax.plot(data['degrees'], data['iterations'], 'o-', label=alg_name, linewidth=2, markersize=6)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Number of Iterations')
    ax.set_title('Iterations vs Degree')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Computation time vs degree
    ax = axes[1, 0]
    for alg_name, data in alg_data.items():
        ax.plot(data['degrees'], data['times'], 'o-', label=alg_name, linewidth=2, markersize=6)
    ax.set_xlabel('Polynomial Degree')
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Estimated Time (s)')
    ax.set_title('Time vs Degree')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Condition number vs degree
    ax = axes[1, 1]
    degrees_list = []
    cond_numbers = []
    for degree in degrees:
        problem = SineApproximationProblem(degree=degree, num_samples=100)
        degrees_list.append(degree)
        cond_numbers.append(problem.condition_number)
    
    ax.semilogy(degrees_list, cond_numbers, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Condition Number')
    ax.set_title('Problem Condition Number vs Degree')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/task2_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/task2_performance_analysis.png")
    plt.show()
    
    # Plot 2: Lambda sensitivity analysis
    degree = 10
    problem = SineApproximationProblem(degree=degree, num_samples=100)
    lambda_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    lambda_results = []
    for lambda_val in lambda_values:
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
        
        lambda_results.append({
            'lambda': lambda_val, 'objective': obj, 'l1_norm': l1_norm,
            'nonzero': nonzero, 'rmse': rmse, 'solution': x
        })
    
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
    plt.savefig('results/plots/task2_lambda_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/task2_lambda_analysis.png")
    plt.show()
    
    # Plot 3: Function approximations for different degrees
    selected_degrees = [3, 7, 11, 15]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, degree in enumerate(selected_degrees):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        problem = SineApproximationProblem(degree=degree, num_samples=100)
        t_fine = np.linspace(-2*np.pi, 2*np.pi, 1000)
        y_true = np.sin(t_fine)
        
        # Plot different algorithm results
        for alg_name, result in all_results[degree].items():
            y_approx = problem.evaluate_polynomial(result['solution'], t_fine)
            ax.plot(t_fine, y_approx, '--', label=alg_name, linewidth=2)
        
        ax.plot(t_fine, y_true, 'k-', label='True sin(t)', linewidth=3)
        ax.plot(problem.sample_points, problem.target_values, 'ko', markersize=3)
        
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_title(f'Degree {degree} Approximations')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/task2_approximation_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/task2_approximation_comparison.png")
    plt.show()
    
    # Save Task 2 data
    task2_data = {
        'all_results': all_results,
        'lambda_analysis': lambda_results,
        'degrees_tested': degrees,
        'lambda_values_tested': lambda_values,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('results/data/task2_results.pkl', 'wb') as f:
        pickle.dump(task2_data, f)
    
    # Save summary data as JSON
    json_data = {
        'degrees_tested': degrees,
        'lambda_values_tested': lambda_values,
        'summary_by_degree': {},
        'lambda_analysis': []
    }
    
    for degree in degrees:
        json_data['summary_by_degree'][str(degree)] = {}
        for alg_name, result in all_results[degree].items():
            json_data['summary_by_degree'][str(degree)][alg_name] = {
                'final_objective': float(result['final_objective']),
                'iterations': int(result['iterations']),
                'l1_norm': float(np.sum(np.abs(result['solution']))),
                'non_zero_coeffs': int(np.sum(np.abs(result['solution']) > 1e-10))
            }
    
    for result in lambda_results:
        json_data['lambda_analysis'].append({
            'lambda': float(result['lambda']),
            'objective': float(result['objective']),
            'l1_norm': float(result['l1_norm']),
            'nonzero': int(result['nonzero']),
            'rmse': float(result['rmse'])
        })
    
    with open('results/data/task2_results.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print("Saved Task 2 data to results/data/")
    
    return task2_data


def main():
    """Generate all images and save data."""
    # Ensure directories exist
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)
    
    print("=" * 80)
    print("GENERATING COMPREHENSIVE IMAGES AND DATA FOR TASKS 1 & 2")
    print("=" * 80)
    
    try:
        # Generate Task 1 images and data
        print("\n" + "=" * 50)
        print("TASK 1: Forward-Backward and Projected Gradient")
        print("=" * 50)
        task1_data = generate_task1_images()
        
        # Generate Task 2 images and data
        print("\n" + "=" * 50)
        print("TASK 2: Active-Set Method and Comprehensive Analysis")
        print("=" * 50)
        task2_data = generate_task2_images()
        
        print("\n" + "=" * 80)
        print("ALL IMAGES AND DATA GENERATED SUCCESSFULLY!")
        print("=" * 80)
        print("Images saved in: results/plots/")
        print("Data saved in: results/data/")
        print("Files generated:")
        
        # List generated files
        plot_files = [f for f in os.listdir('results/plots') if f.endswith('.png')]
        data_files = [f for f in os.listdir('results/data') if f.endswith(('.pkl', '.json'))]
        
        print(f"  Plots ({len(plot_files)}):")
        for f in sorted(plot_files):
            print(f"    - {f}")
        
        print(f"  Data ({len(data_files)}):")
        for f in sorted(data_files):
            print(f"    - {f}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
