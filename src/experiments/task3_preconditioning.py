"""
Task 3: Condition Number Analysis and Pre-conditioning.

This script analyzes the condition number growth patterns and implements
various pre-conditioning strategies to improve numerical stability.
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
from problems.preconditioned_problem import PreconditionedSineProblem
from algorithms.forward_backward import ForwardBackward
from algorithms.projected_gradient import ProjectedGradient
from algorithms.active_set_method import ActiveSetMethod


def analyze_condition_number_growth():
    """Analyze how condition number grows with polynomial degree."""
    print("=" * 80)
    print("TASK 3: CONDITION NUMBER ANALYSIS")
    print("=" * 80)
    
    degrees = list(range(3, 21))  # Test up to degree 20
    condition_numbers = []
    singular_value_ratios = []
    
    print("Analyzing condition number growth...")
    
    for degree in degrees:
        problem = SineApproximationProblem(degree=degree, num_samples=100)
        condition_numbers.append(problem.condition_number)
        
        # Compute singular value ratio (largest / smallest)
        ATA = problem.vandermonde_matrix.T @ problem.vandermonde_matrix
        singular_values = np.linalg.svd(ATA, compute_uv=False)
        singular_value_ratios.append(singular_values[0] / singular_values[-1])
        
        print(f"Degree {degree:2d}: κ = {problem.condition_number:.2e}")
    
    # Plot condition number growth
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Condition number vs degree
    ax = axes[0, 0]
    ax.semilogy(degrees, condition_numbers, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Condition Number κ(A^T A)')
    ax.set_title('Condition Number Growth with Polynomial Degree')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log-log plot to check growth rate
    ax = axes[0, 1]
    ax.loglog(degrees, condition_numbers, 'ro-', linewidth=2, markersize=6)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Condition Number κ(A^T A)')
    ax.set_title('Log-Log Plot: Growth Rate Analysis')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Growth rate analysis
    ax = axes[1, 0]
    # Compute growth rate (ratio of consecutive condition numbers)
    growth_rates = []
    for i in range(1, len(condition_numbers)):
        rate = condition_numbers[i] / condition_numbers[i-1]
        growth_rates.append(rate)
    
    ax.plot(degrees[1:], growth_rates, 'go-', linewidth=2, markersize=6)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Growth Rate (κ_{i+1} / κ_i)')
    ax.set_title('Condition Number Growth Rate')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Singular value distribution for selected degrees
    ax = axes[1, 1]
    selected_degrees = [5, 10, 15, 20]
    for degree in selected_degrees:
        if degree <= max(degrees):
            problem = SineApproximationProblem(degree=degree, num_samples=100)
            ATA = problem.vandermonde_matrix.T @ problem.vandermonde_matrix
            singular_values = np.linalg.svd(ATA, compute_uv=False)
            ax.semilogy(range(1, len(singular_values) + 1), singular_values, 
                       'o-', label=f'Degree {degree}', linewidth=2, markersize=4)
    
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Value Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/task3_condition_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/task3_condition_analysis.png")
    plt.show()
    
    return {
        'degrees': degrees,
        'condition_numbers': condition_numbers,
        'singular_value_ratios': singular_value_ratios,
        'growth_rates': growth_rates
    }


def test_preconditioning_methods():
    """Test different pre-conditioning methods on high-degree problems."""
    print("\n" + "=" * 80)
    print("TESTING PRE-CONDITIONING METHODS")
    print("=" * 80)
    
    # Test on a high-degree problem where conditioning is poor
    test_degree = 15
    original_problem = SineApproximationProblem(degree=test_degree, num_samples=100)
    
    print(f"Testing pre-conditioning on degree {test_degree} problem")
    print(f"Original condition number: {original_problem.condition_number:.2e}")
    
    # Test different pre-conditioning methods
    methods = ['diagonal', 'svd', 'cholesky']
    preconditioned_problems = {}
    
    for method in methods:
        print(f"\nTesting {method} pre-conditioning...")
        try:
            prec_problem = PreconditionedSineProblem(original_problem, method)
            preconditioned_problems[method] = prec_problem
            
            summary = prec_problem.get_preconditioning_summary()
            print(f"  Success: {summary['method']}")
            print(f"  New condition number: {summary['preconditioned_condition_number']:.2e}")
            print(f"  Improvement factor: {summary['condition_improvement']:.2e}")
            print(f"  Log10 improvement: {summary['condition_improvement_log10']:.2f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Generate conditioning comparison plots
    for method, prec_problem in preconditioned_problems.items():
        prec_problem.plot_conditioning_comparison()
    
    return preconditioned_problems


def compare_algorithm_performance():
    """Compare algorithm performance on original vs pre-conditioned problems."""
    print("\n" + "=" * 80)
    print("COMPARING ALGORITHM PERFORMANCE")
    print("=" * 80)
    
    # Test on a medium-degree problem where pre-conditioning should help
    test_degree = 12
    original_problem = SineApproximationProblem(degree=test_degree, num_samples=100)
    
    print(f"Testing algorithms on degree {test_degree} problem")
    print(f"Original condition number: {original_problem.condition_number:.2e}")
    
    # Test algorithms on original problem
    print("\nTesting algorithms on ORIGINAL problem:")
    original_results = {}
    
    # Test PG
    try:
        pg = ProjectedGradient(original_problem)
        pg_result = pg.solve(max_iterations=10000, tolerance=1e-6, verbose=False)
        original_results['PG'] = pg_result
        print(f"  PG: obj={pg_result['final_objective']:.6f}, iter={pg_result['iterations']}")
    except Exception as e:
        print(f"  PG: FAILED - {e}")
    
    # Test ASM
    try:
        asm = ActiveSetMethod(original_problem)
        asm_result = asm.solve(max_iterations=1000, tolerance=1e-6, verbose=False)
        original_results['ASM'] = asm_result
        print(f"  ASM: obj={asm_result['final_objective']:.6f}, iter={asm_result['iterations']}")
    except Exception as e:
        print(f"  ASM: FAILED - {e}")
    
    # Test FB
    try:
        fb = ForwardBackward(original_problem, lambda_param=1.0)
        fb_result = fb.solve(max_iterations=10000, tolerance=1e-6, verbose=False)
        original_results['FB'] = fb_result
        print(f"  FB: obj={fb_result['final_objective']:.6f}, iter={fb_result['iterations']}")
    except Exception as e:
        print(f"  FB: FAILED - {e}")
    
    # Test algorithms on pre-conditioned problem
    print("\nTesting algorithms on PRE-CONDITIONED problem (diagonal):")
    prec_problem = PreconditionedSineProblem(original_problem, 'diagonal')
    print(f"Pre-conditioned condition number: {prec_problem.condition_number:.2e}")
    print(f"Improvement factor: {prec_problem.get_preconditioning_summary()['condition_improvement']:.2e}")
    
    preconditioned_results = {}
    
    # Test PG on pre-conditioned problem
    try:
        pg_prec = ProjectedGradient(prec_problem)
        pg_prec_result = pg_prec.solve(max_iterations=10000, tolerance=1e-6, verbose=False)
        preconditioned_results['PG'] = pg_prec_result
        print(f"  PG: obj={pg_prec_result['final_objective']:.6f}, iter={pg_prec_result['iterations']}")
    except Exception as e:
        print(f"  PG: FAILED - {e}")
    
    # Test ASM on pre-conditioned problem
    try:
        asm_prec = ActiveSetMethod(prec_problem)
        asm_prec_result = asm_prec.solve(max_iterations=1000, tolerance=1e-6, verbose=False)
        preconditioned_results['ASM'] = asm_prec_result
        print(f"  ASM: obj={asm_prec_result['final_objective']:.6f}, iter={asm_prec_result['iterations']}")
    except Exception as e:
        print(f"  ASM: FAILED - {e}")
    
    # Test FB on pre-conditioned problem
    try:
        fb_prec = ForwardBackward(prec_problem, lambda_param=1.0)
        fb_prec_result = fb_prec.solve(max_iterations=10000, tolerance=1e-6, verbose=False)
        preconditioned_results['FB'] = fb_prec_result
        print(f"  FB: obj={fb_prec_result['final_objective']:.6f}, iter={fb_prec_result['iterations']}")
    except Exception as e:
        print(f"  FB: FAILED - {e}")
    
    # Compare solutions
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    comparison_data = {
        'original': original_results,
        'preconditioned': preconditioned_results,
        'problem_info': {
            'degree': test_degree,
            'original_condition': original_problem.condition_number,
            'preconditioned_condition': prec_problem.condition_number,
            'improvement_factor': prec_problem.get_preconditioning_summary()['condition_improvement']
        }
    }
    
    # Create comparison plot
    if original_results and preconditioned_results:
        create_performance_comparison_plot(original_results, preconditioned_results, test_degree)
    
    return comparison_data


def create_performance_comparison_plot(original_results, preconditioned_results, degree):
    """Create a performance comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data - only include algorithms that exist in both results
    algorithms = [alg for alg in original_results.keys() if alg in preconditioned_results]
    
    if not algorithms:
        print("No common algorithms found for comparison")
        return
    
    # Plot 1: Objective values comparison
    ax = axes[0, 0]
    x = np.arange(len(algorithms))
    width = 0.35
    
    orig_objectives = [original_results[alg]['final_objective'] for alg in algorithms]
    prec_objectives = [preconditioned_results[alg]['final_objective'] for alg in algorithms]
    
    bars1 = ax.bar(x - width/2, orig_objectives, width, label='Original Problem', alpha=0.8)
    bars2 = ax.bar(x + width/2, prec_objectives, width, label='Pre-conditioned Problem', alpha=0.8)
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Final Objective Value')
    ax.set_title(f'Objective Values Comparison (Degree {degree})')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 2: Iterations comparison
    ax = axes[0, 1]
    orig_iterations = [original_results[alg]['iterations'] for alg in algorithms]
    prec_iterations = [preconditioned_results[alg]['iterations'] for alg in algorithms]
    
    bars1 = ax.bar(x - width/2, orig_iterations, width, label='Original Problem', alpha=0.8)
    bars2 = ax.bar(x + width/2, prec_iterations, width, label='Pre-conditioned Problem', alpha=0.8)
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Number of Iterations')
    ax.set_title(f'Iterations Comparison (Degree {degree})')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
    
    # Plot 3: Solution quality comparison (L1 norm)
    ax = axes[1, 0]
    orig_l1_norms = []
    prec_l1_norms = []
    
    for alg in algorithms:
        if 'solution' in original_results[alg]:
            orig_l1 = np.sum(np.abs(original_results[alg]['solution']))
            orig_l1_norms.append(orig_l1)
        else:
            orig_l1_norms.append(0)
        
        if 'solution' in preconditioned_results[alg]:
            prec_l1 = np.sum(np.abs(preconditioned_results[alg]['solution']))
            prec_l1_norms.append(prec_l1)
        else:
            prec_l1_norms.append(0)
    
    bars1 = ax.bar(x - width/2, orig_l1_norms, width, label='Original Problem', alpha=0.8)
    bars2 = ax.bar(x + width/2, prec_l1_norms, width, label='Pre-conditioned Problem', alpha=0.8)
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('L1 Norm of Solution')
    ax.set_title(f'Solution Sparsity Comparison (Degree {degree})')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Condition number comparison
    ax = axes[1, 1]
    condition_numbers = [1.0, 1.0]  # Placeholder for condition numbers
    labels = ['Original', 'Pre-conditioned']
    
    bars = ax.bar(labels, condition_numbers, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Condition Number')
    ax.set_title('Condition Number Comparison')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/plots/task3_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/task3_performance_comparison.png")
    plt.show()


def comprehensive_preconditioning_analysis():
    """Perform comprehensive analysis of pre-conditioning effectiveness."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PRE-CONDITIONING ANALYSIS")
    print("=" * 80)
    
    degrees = [8, 10, 12, 15]
    methods = ['diagonal', 'svd']
    results = {}
    
    for degree in degrees:
        print(f"\nAnalyzing degree {degree}...")
        original_problem = SineApproximationProblem(degree=degree, num_samples=100)
        
        degree_results = {
            'original_condition': original_problem.condition_number,
            'methods': {}
        }
        
        for method in methods:
            try:
                prec_problem = PreconditionedSineProblem(original_problem, method)
                summary = prec_problem.get_preconditioning_summary()
                
                degree_results['methods'][method] = {
                    'condition_number': summary['preconditioned_condition_number'],
                    'improvement_factor': summary['condition_improvement'],
                    'log10_improvement': summary['condition_improvement_log10']
                }
                
                print(f"  {method}: κ = {summary['preconditioned_condition_number']:.2e}, "
                      f"improvement = {summary['condition_improvement']:.2e}")
                
            except Exception as e:
                print(f"  {method}: FAILED - {e}")
                degree_results['methods'][method] = None
        
        results[degree] = degree_results
    
    # Create comprehensive analysis plot
    create_comprehensive_analysis_plot(results)
    
    return results


def create_comprehensive_analysis_plot(results):
    """Create comprehensive pre-conditioning analysis plot."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    degrees = list(results.keys())
    
    # Plot 1: Condition numbers comparison
    ax = axes[0, 0]
    x = np.arange(len(degrees))
    width = 0.25
    
    # Original condition numbers
    orig_conditions = [results[d]['original_condition'] for d in degrees]
    ax.bar(x - width, orig_conditions, width, label='Original', color='blue', alpha=0.7)
    
    # Pre-conditioned condition numbers
    for i, method in enumerate(['diagonal', 'svd']):
        conditions = []
        for degree in degrees:
            if results[degree]['methods'][method]:
                conditions.append(results[degree]['methods'][method]['condition_number'])
            else:
                conditions.append(0)
        
        color = 'red' if method == 'diagonal' else 'green'
        ax.bar(x + i*width, conditions, width, label=f'{method.capitalize()}', 
               color=color, alpha=0.7)
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Condition Number')
    ax.set_title('Condition Number Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(degrees)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Improvement factors
    ax = axes[0, 1]
    for method in ['diagonal', 'svd']:
        improvements = []
        for degree in degrees:
            if results[degree]['methods'][method]:
                improvements.append(results[degree]['methods'][method]['improvement_factor'])
            else:
                improvements.append(1)
        
        color = 'red' if method == 'diagonal' else 'green'
        ax.plot(degrees, improvements, 'o-', label=f'{method.capitalize()}', 
               color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Improvement Factor')
    ax.set_title('Pre-conditioning Effectiveness')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Log10 improvement
    ax = axes[1, 0]
    for method in ['diagonal', 'svd']:
        log_improvements = []
        for degree in degrees:
            if results[degree]['methods'][method]:
                log_improvements.append(results[degree]['methods'][method]['log10_improvement'])
            else:
                log_improvements.append(0)
        
        color = 'red' if method == 'diagonal' else 'green'
        ax.plot(degrees, log_improvements, 'o-', label=f'{method.capitalize()}', 
               color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Log10(Improvement Factor)')
    ax.set_title('Logarithmic Improvement Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    methods = ['diagonal', 'svd']
    avg_improvements = []
    
    for method in methods:
        improvements = []
        for degree in degrees:
            if results[degree]['methods'][method]:
                improvements.append(results[degree]['methods'][method]['improvement_factor'])
        
        if improvements:
            avg_improvements.append(np.mean(improvements))
        else:
            avg_improvements.append(1)
    
    bars = ax.bar(methods, avg_improvements, color=['red', 'green'], alpha=0.7)
    ax.set_ylabel('Average Improvement Factor')
    ax.set_title('Average Pre-conditioning Effectiveness')
    ax.set_yscale('log')
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2e}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/plots/task3_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/task3_comprehensive_analysis.png")
    plt.show()


def main():
    """Main function for Task 3 experiments."""
    # Ensure directories exist
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)
    
    print("=" * 80)
    print("TASK 3: CONDITION NUMBER ANALYSIS AND PRE-CONDITIONING")
    print("=" * 80)
    
    try:
        # 1. Analyze condition number growth
        condition_analysis = analyze_condition_number_growth()
        
        # 2. Test pre-conditioning methods
        preconditioned_problems = test_preconditioning_methods()
        
        # 3. Compare algorithm performance
        performance_comparison = compare_algorithm_performance()
        
        # 4. Comprehensive analysis
        comprehensive_analysis = comprehensive_preconditioning_analysis()
        
        # Save all results
        task3_data = {
            'condition_analysis': condition_analysis,
            'preconditioned_problems': {k: v.get_preconditioning_summary() 
                                      for k, v in preconditioned_problems.items()},
            'performance_comparison': performance_comparison,
            'comprehensive_analysis': comprehensive_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save as pickle
        with open('results/data/task3_results.pkl', 'wb') as f:
            pickle.dump(task3_data, f)
        
        # Save summary as JSON
        with open('results/data/task3_results.json', 'w') as f:
            # Helper to convert numpy types recursively
            def to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                # Handle numpy scalars
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, dict):
                    return {k: to_serializable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [to_serializable(v) for v in obj]
                return obj

            json_data = {}
            for key, value in task3_data.items():
                if key == 'condition_analysis':
                    json_data[key] = {
                        'degrees': [int(d) for d in value['degrees']],
                        'condition_numbers': [float(c) for c in value['condition_numbers']],
                        'growth_rates': [float(r) for r in value['growth_rates']]
                    }
                elif key == 'preconditioned_problems':
                    json_data[key] = to_serializable(value)
                elif key == 'performance_comparison':
                    json_data[key] = to_serializable(value)
                elif key == 'comprehensive_analysis':
                    json_data[key] = to_serializable(value)
                else:
                    json_data[key] = to_serializable(value)

            json.dump(json_data, f, indent=2)
        
        print("\n" + "=" * 80)
        print("TASK 3 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Generated files:")
        print("  - results/plots/task3_condition_analysis.png")
        print("  - results/plots/task3_performance_comparison.png")
        print("  - results/plots/task3_comprehensive_analysis.png")
        print("  - results/data/task3_results.pkl")
        print("  - results/data/task3_results.json")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
