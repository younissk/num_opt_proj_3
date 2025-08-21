"""
Task 2 Experiments: Comprehensive testing of FB, PG, and ASM algorithms.

This script runs experiments with different polynomial degrees (up to 15),
starting points, and regularization parameters to compare the three algorithms.
"""

from algorithms.active_set_method import ActiveSetMethod
from algorithms.projected_gradient import ProjectedGradient
from algorithms.forward_backward import ForwardBackward
from problems.sine_approximation import SineApproximationProblem
import time
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class Task2Experiments:
    """Comprehensive experiment runner for Task 2."""

    def __init__(self):
        self.results = {}
        self.algorithms = ['FB', 'PG', 'ASM']

    def run_single_experiment(self,
                              algorithm: str,
                              problem: SineApproximationProblem,
                              lambda_param: Optional[float] = None,
                              x0: Optional[np.ndarray] = None,
                              max_iterations: int = 50000,
                              tolerance: float = 1e-6) -> Dict:
        """Run a single experiment with specified parameters."""

        start_time = time.time()

        if algorithm == 'FB':
            if lambda_param is None:
                lambda_param = 1.0
            solver = ForwardBackward(problem, lambda_param)
            result = solver.solve(x0=x0, max_iterations=max_iterations,
                                  tolerance=tolerance, verbose=False)
        elif algorithm == 'PG':
            solver = ProjectedGradient(problem)
            result = solver.solve(x0=x0, max_iterations=max_iterations,
                                  tolerance=tolerance, verbose=False)
        elif algorithm == 'ASM':
            solver = ActiveSetMethod(problem)
            result = solver.solve(x0=x0, max_iterations=max_iterations,
                                  tolerance=tolerance, verbose=False)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        elapsed_time = time.time() - start_time

        # Extract key metrics
        x_solution = result['solution']
        final_objective = result['final_objective']
        iterations = result['iterations']

        # Compute additional metrics
        l1_norm = np.sum(np.abs(x_solution))
        non_zero_coeffs = np.sum(np.abs(x_solution) > 1e-10)

        # Get optimality measure based on algorithm
        if algorithm == 'FB':
            optimality_measure = result['final_proximal_residual']
            optimality_name = 'proximal_residual'
        else:  # PG or ASM
            optimality_measure = result['final_projected_gradient']
            optimality_name = 'projected_gradient'

        # Compute approximation error
        t_test = np.linspace(-2*np.pi, 2*np.pi, 1000)
        y_true = np.sin(t_test)
        y_approx = problem.evaluate_polynomial(x_solution, t_test)
        approximation_error = np.sqrt(np.mean((y_true - y_approx)**2))
        max_approximation_error = np.max(np.abs(y_true - y_approx))

        return {
            'algorithm': algorithm,
            'degree': problem.degree,
            'lambda': lambda_param if algorithm == 'FB' else None,
            'final_objective': final_objective,
            'iterations': iterations,
            'optimality_measure': optimality_measure,
            'optimality_name': optimality_name,
            'l1_norm': l1_norm,
            'non_zero_coeffs': non_zero_coeffs,
            'approximation_error': approximation_error,
            'max_approximation_error': max_approximation_error,
            'solution': x_solution,
            'elapsed_time': elapsed_time,
            'convergence_history': result['convergence_history'],
            'problem_condition_number': problem.condition_number
        }

    def experiment_degree_comparison(self) -> Dict:
        """Experiment 1: Compare algorithms across different polynomial degrees."""
        print("=" * 80)
        print("EXPERIMENT 1: Polynomial Degree Comparison (n = 3 to 15)")
        print("=" * 80)

        degrees = list(range(3, 16))  # n = 3 to 15
        lambda_values = [0.1, 0.5, 1.0]  # For FB algorithm
        results = []

        for degree in degrees:
            print(f"\nTesting degree {degree}...")
            problem = SineApproximationProblem(degree=degree, num_samples=100)

            # Test PG and ASM (no lambda parameter)
            for algorithm in ['PG', 'ASM']:
                try:
                    result = self.run_single_experiment(algorithm, problem)
                    results.append(result)
                    print(f"  {algorithm}: obj={result['final_objective']:.6f}, "
                          f"iter={result['iterations']}, opt={result['optimality_measure']:.2e}")
                except Exception as e:
                    print(f"  {algorithm}: FAILED - {e}")

            # Test FB with different lambda values
            for lambda_val in lambda_values:
                try:
                    result = self.run_single_experiment(
                        'FB', problem, lambda_param=lambda_val)
                    results.append(result)
                    print(f"  FB(λ={lambda_val}): obj={result['final_objective']:.6f}, "
                          f"iter={result['iterations']}, opt={result['optimality_measure']:.2e}")
                except Exception as e:
                    print(f"  FB(λ={lambda_val}): FAILED - {e}")

        return {'degree_comparison': results}

    def experiment_lambda_sensitivity(self) -> Dict:
        """Experiment 2: Test FB algorithm sensitivity to lambda parameter."""
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: Lambda Parameter Sensitivity (FB Algorithm)")
        print("=" * 80)

        degree = 10  # Fixed degree
        lambda_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        problem = SineApproximationProblem(degree=degree, num_samples=100)
        results = []

        print(f"Testing degree {degree} with various λ values...")

        for lambda_val in lambda_values:
            try:
                result = self.run_single_experiment(
                    'FB', problem, lambda_param=lambda_val)
                results.append(result)
                print(f"  λ={lambda_val:5.2f}: obj={result['final_objective']:.6f}, "
                      f"L1={result['l1_norm']:.4f}, nonzero={result['non_zero_coeffs']}")
            except Exception as e:
                print(f"  λ={lambda_val:5.2f}: FAILED - {e}")

        return {'lambda_sensitivity': results}

    def experiment_starting_points(self) -> Dict:
        """Experiment 3: Test different starting points."""
        print("\n" + "=" * 80)
        print("EXPERIMENT 3: Different Starting Points")
        print("=" * 80)

        degree = 8
        problem = SineApproximationProblem(degree=degree, num_samples=100)

        # Different starting points
        starting_points = {
            'zero': np.zeros(degree + 1),
            'random_small': np.random.randn(degree + 1) * 0.1,
            'random_large': np.random.randn(degree + 1) * 0.5,
            # Feasible for L1 ball
            'uniform': np.ones(degree + 1) / (degree + 1),
            'alternating': np.array([(-1)**(i) * 0.1 for i in range(degree + 1)])
        }

        results = []

        for start_name, x0 in starting_points.items():
            print(f"\nTesting starting point: {start_name}")
            print(f"  Initial L1 norm: {np.sum(np.abs(x0)):.6f}")

            # Test all algorithms
            for algorithm in self.algorithms:
                try:
                    if algorithm == 'FB':
                        result = self.run_single_experiment(algorithm, problem,
                                                            lambda_param=1.0, x0=x0)
                    else:
                        result = self.run_single_experiment(
                            algorithm, problem, x0=x0)

                    result['starting_point'] = start_name
                    results.append(result)
                    print(f"    {algorithm}: obj={result['final_objective']:.6f}, "
                          f"iter={result['iterations']}")
                except Exception as e:
                    print(f"    {algorithm}: FAILED - {e}")

        return {'starting_points': results}

    def create_comparison_plots(self, results: Dict) -> None:
        """Create comprehensive comparison plots."""
        print("\n" + "=" * 80)
        print("CREATING COMPARISON PLOTS")
        print("=" * 80)

        # Plot 1: Degree vs Performance
        if 'degree_comparison' in results:
            self._plot_degree_comparison(results['degree_comparison'])

        # Plot 2: Lambda sensitivity
        if 'lambda_sensitivity' in results:
            self._plot_lambda_sensitivity(results['lambda_sensitivity'])

        # Plot 3: Approximation quality
        if 'degree_comparison' in results:
            self._plot_approximation_quality(results['degree_comparison'])

        # Plot 4: Convergence histories
        if 'starting_points' in results:
            self._plot_convergence_comparison(results['starting_points'])

    def _plot_degree_comparison(self, results: List[Dict]) -> None:
        """Plot performance vs polynomial degree."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Organize data by algorithm
        data_by_alg = {}
        for result in results:
            alg = result['algorithm']
            if result['lambda'] is not None:
                alg += f"(λ={result['lambda']})"

            if alg not in data_by_alg:
                data_by_alg[alg] = {
                    'degrees': [], 'objectives': [], 'iterations': [], 'errors': []}

            data_by_alg[alg]['degrees'].append(result['degree'])
            data_by_alg[alg]['objectives'].append(result['final_objective'])
            data_by_alg[alg]['iterations'].append(result['iterations'])
            data_by_alg[alg]['errors'].append(result['approximation_error'])

        # Plot 1: Objective values
        ax = axes[0, 0]
        for alg, data in data_by_alg.items():
            ax.plot(data['degrees'], data['objectives'], 'o-',
                    label=alg, linewidth=2, markersize=6)
        ax.set_xlabel('Polynomial Degree')
        ax.set_ylabel('Final Objective Value')
        ax.set_title('Objective Value vs Degree')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Iterations
        ax = axes[0, 1]
        for alg, data in data_by_alg.items():
            ax.plot(data['degrees'], data['iterations'], 'o-',
                    label=alg, linewidth=2, markersize=6)
        ax.set_xlabel('Polynomial Degree')
        ax.set_ylabel('Number of Iterations')
        ax.set_title('Iterations vs Degree')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Approximation error
        ax = axes[1, 0]
        for alg, data in data_by_alg.items():
            ax.semilogy(data['degrees'], data['errors'], 'o-',
                        label=alg, linewidth=2, markersize=6)
        ax.set_xlabel('Polynomial Degree')
        ax.set_ylabel('RMSE Approximation Error')
        ax.set_title('Approximation Quality vs Degree')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Condition number
        ax = axes[1, 1]
        degrees = []
        condition_numbers = []
        for result in results:
            if result['algorithm'] == 'PG':  # Use PG as representative
                degrees.append(result['degree'])
                condition_numbers.append(result['problem_condition_number'])

        ax.semilogy(degrees, condition_numbers,
                    'ro-', linewidth=2, markersize=6)
        ax.set_xlabel('Polynomial Degree')
        ax.set_ylabel('Condition Number')
        ax.set_title('Problem Condition Number vs Degree')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/plots/task2_degree_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_lambda_sensitivity(self, results: List[Dict]) -> None:
        """Plot FB algorithm sensitivity to lambda."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        lambdas = [r['lambda'] for r in results]
        objectives = [r['final_objective'] for r in results]
        l1_norms = [r['l1_norm'] for r in results]
        non_zeros = [r['non_zero_coeffs'] for r in results]
        errors = [r['approximation_error'] for r in results]

        # Plot 1: Objective vs lambda
        axes[0, 0].semilogx(lambdas, objectives, 'bo-',
                            linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Regularization Parameter λ')
        axes[0, 0].set_ylabel('Final Objective Value')
        axes[0, 0].set_title('Objective Value vs λ')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: L1 norm vs lambda
        axes[0, 1].semilogx(lambdas, l1_norms, 'ro-',
                            linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Regularization Parameter λ')
        axes[0, 1].set_ylabel('L1 Norm of Solution')
        axes[0, 1].set_title('Sparsity vs λ')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Number of non-zeros vs lambda
        axes[1, 0].semilogx(lambdas, non_zeros, 'go-',
                            linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Regularization Parameter λ')
        axes[1, 0].set_ylabel('Number of Non-zero Coefficients')
        axes[1, 0].set_title('Active Coefficients vs λ')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Approximation error vs lambda
        axes[1, 1].loglog(lambdas, errors, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Regularization Parameter λ')
        axes[1, 1].set_ylabel('RMSE Approximation Error')
        axes[1, 1].set_title('Approximation Quality vs λ')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/plots/task2_lambda_sensitivity.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_approximation_quality(self, results: List[Dict]) -> None:
        """Plot function approximations for different methods."""
        # Find some representative results
        representative_results = {}
        for result in results:
            if result['degree'] == 10:  # Focus on degree 10
                key = result['algorithm']
                if result['lambda'] is not None:
                    key += f"(λ={result['lambda']})"
                if key not in representative_results:
                    representative_results[key] = result

        if not representative_results:
            return

        # Create plots
        fig, axes = plt.subplots(2, len(representative_results), figsize=(
            5*len(representative_results), 10))
        if len(representative_results) == 1:
            axes = axes.reshape(2, 1)

        t_fine = np.linspace(-2*np.pi, 2*np.pi, 1000)
        y_true = np.sin(t_fine)

        for i, (method, result) in enumerate(representative_results.items()):
            problem = SineApproximationProblem(
                degree=result['degree'], num_samples=100)
            y_approx = problem.evaluate_polynomial(result['solution'], t_fine)

            # Top plot: Function comparison
            axes[0, i].plot(t_fine, y_true, 'b-',
                            label='True sin(t)', linewidth=2)
            axes[0, i].plot(t_fine, y_approx, 'r--',
                            label=f'{method}', linewidth=2)
            axes[0, i].plot(problem.sample_points,
                            problem.target_values, 'ko', markersize=3)
            axes[0, i].set_xlabel('t')
            axes[0, i].set_ylabel('y')
            axes[0, i].set_title(
                f'{method}\nRMSE = {result["approximation_error"]:.4f}')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)

            # Bottom plot: Coefficients
            coeffs = result['solution']
            axes[1, i].bar(range(len(coeffs)), coeffs, alpha=0.7)
            axes[1, i].set_xlabel('Coefficient Index')
            axes[1, i].set_ylabel('Coefficient Value')
            axes[1, i].set_title(
                f'Coefficients (L1 norm = {result["l1_norm"]:.4f})')
            axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/plots/task2_approximation_quality.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_convergence_comparison(self, results: List[Dict]) -> None:
        """Plot convergence histories for different starting points."""
        # Group by algorithm and starting point
        data_by_start = {}
        for result in results:
            if 'starting_point' in result:
                start = result['starting_point']
                if start not in data_by_start:
                    data_by_start[start] = {}
                data_by_start[start][result['algorithm']] = result

        # Plot convergence for zero starting point (most common)
        if 'zero' in data_by_start:
            plt.figure(figsize=(12, 8))

            for alg, result in data_by_start['zero'].items():
                history = result['convergence_history']
                if alg == 'FB':
                    plt.semilogy(history['iterations'], history['proximal_residuals'],
                                 'o-', label=f"{alg}: Proximal Residual", linewidth=2)
                else:
                    plt.semilogy(history['iterations'], history['projected_gradients'],
                                 'o-', label=f"{alg}: Projected Gradient", linewidth=2)

            plt.xlabel('Iteration')
            plt.ylabel('Optimality Measure')
            plt.title('Convergence Comparison (Zero Starting Point)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('results/plots/task2_convergence.png',
                        dpi=300, bbox_inches='tight')
            plt.show()

    def run_all_experiments(self) -> Dict:
        """Run all Task 2 experiments."""
        print("Starting Task 2 Comprehensive Experiments...")

        all_results = {}

        # Run experiments
        all_results.update(self.experiment_degree_comparison())
        all_results.update(self.experiment_lambda_sensitivity())
        all_results.update(self.experiment_starting_points())

        # Create plots
        self.create_comparison_plots(all_results)

        # Generate summary report
        self.generate_summary_report(all_results)

        return all_results

    def generate_summary_report(self, results: Dict) -> None:
        """Generate a summary report of all experiments."""
        print("\n" + "=" * 80)
        print("TASK 2 SUMMARY REPORT")
        print("=" * 80)

        # Best performance by algorithm
        if 'degree_comparison' in results:
            print("\nBest Performance by Algorithm (Degree 10):")
            degree_10_results = [
                r for r in results['degree_comparison'] if r['degree'] == 10]

            for alg in ['PG', 'ASM']:
                alg_results = [
                    r for r in degree_10_results if r['algorithm'] == alg]
                if alg_results:
                    best = min(
                        alg_results, key=lambda x: x['approximation_error'])
                    print(f"  {alg}: RMSE = {best['approximation_error']:.6f}, "
                          f"Iter = {best['iterations']}, Obj = {best['final_objective']:.6f}")

            # FB results with different lambdas
            fb_results = [
                r for r in degree_10_results if r['algorithm'] == 'FB']
            if fb_results:
                best_fb = min(
                    fb_results, key=lambda x: x['approximation_error'])
                print(f"  FB: RMSE = {best_fb['approximation_error']:.6f}, "
                      f"Iter = {best_fb['iterations']}, λ = {best_fb['lambda']}")

        # Sparsity analysis
        if 'lambda_sensitivity' in results:
            print("\nSparsity Analysis (FB Algorithm):")
            lambda_results = results['lambda_sensitivity']
            for result in lambda_results:
                print(f"  λ = {result['lambda']:5.2f}: {result['non_zero_coeffs']} non-zero coeffs, "
                      f"L1 = {result['l1_norm']:.4f}")

        print("\nAll experiments completed successfully!")


def main():
    """Main function to run Task 2 experiments."""
    # Create results directory
    os.makedirs('results/plots', exist_ok=True)

    # Run experiments
    experiments = Task2Experiments()
    results = experiments.run_all_experiments()

    print("\nTask 2 experiments completed!")
    return results


if __name__ == "__main__":
    main()
