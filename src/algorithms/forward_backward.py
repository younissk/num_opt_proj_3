"""
Forward-Backward (Proximal Gradient) algorithm for penalized Lasso.

This implements the FB method for solving:
    min_x (1/2)||Ax - b||² + λ||x||₁

The algorithm alternates between:
1. Forward step: gradient descent on the smooth part
2. Backward step: proximal operator for the L1 penalty
"""

import numpy as np
from typing import Dict, Optional, Any
from problems.base_problem import BaseOptimizationProblem


class ForwardBackward:
    """
    Forward-Backward algorithm for penalized Lasso optimization.

    Algorithm:
        x^{k+1} = prox_{λ||·||₁}(x^k - (1/L)∇f(x^k))

    where:
        - f(x) = (1/2)||Ax - b||² (smooth part)
        - g(x) = λ||x||₁ (non-smooth part)
        - L is the Lipschitz constant of ∇f
    """

    def __init__(self, problem: BaseOptimizationProblem, lambda_param: float):
        """
        Initialize the Forward-Backward algorithm.

        Args:
            problem: Sine approximation problem instance
            lambda_param: Regularization parameter λ > 0
        """
        self.problem = problem
        self.lambda_param = lambda_param
        self.step_size = 1.0 / problem.lipschitz_constant

        # Optimality conditions:
        # For the penalized Lasso: min f(x) + g(x)
        # The optimality condition is: 0 ∈ ∇f(x*) + ∂g(x*)
        # This means: -∇f(x*) ∈ ∂g(x*)
        # Which is equivalent to: x* = prox_{g}(x* - (1/L)∇f(x*))

    def solve(self,
              x0: Optional[np.ndarray] = None,
              max_iterations: int = 100000,
              tolerance: float = 1e-6,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Solve the penalized Lasso problem using Forward-Backward.

        Args:
            x0: Initial guess (default: zero vector)
            max_iterations: Maximum number of iterations
            tolerance: Tolerance for stopping criterion
            verbose: Whether to print progress

        Returns:
            Dictionary containing solution and convergence history
        """
        if x0 is None:
            x0 = np.zeros(self.problem.degree + 1)

        x = x0.copy()
        history = {
            'iterations': [],
            'objective_values': [],
            'proximal_residuals': [],
            'gradient_norms': []
        }

        if verbose:
            print(f"Forward-Backward Algorithm (λ = {self.lambda_param})")
            print(f"Step size: {self.step_size:.2e}")
            print(f"Lipschitz constant: {self.problem.lipschitz_constant:.2e}")
            print("-" * 60)

        for k in range(max_iterations):
            # Store current state
            current_objective = self.problem.objective_function(
                x) + self.lambda_param * np.sum(np.abs(x))
            current_prox_residual = self.problem.compute_proximal_residual(
                x, self.lambda_param)
            current_grad_norm = np.linalg.norm(self.problem.gradient(x))

            history['iterations'].append(k)
            history['objective_values'].append(current_objective)
            history['proximal_residuals'].append(current_prox_residual)
            history['gradient_norms'].append(current_grad_norm)

            # Check stopping criterion
            if current_prox_residual < tolerance:
                if verbose:
                    print(
                        f"Converged at iteration {k}: proximal residual = {current_prox_residual:.2e}")
                break

            # Forward step: gradient descent
            grad_f = self.problem.gradient(x)
            x_temp = x - self.step_size * grad_f

            # Backward step: proximal operator
            x_new = self.problem.l1_proximal_operator(
                x_temp, self.lambda_param * self.step_size)

            # Update iterate
            x = x_new

            # Print progress every 1000 iterations
            if verbose and (k + 1) % 1000 == 0:
                print(f"Iteration {k+1:6d}: obj = {current_objective:.6f}, "
                      f"prox_res = {current_prox_residual:.2e}")

        if verbose:
            print("-" * 60)
            print(f"Final objective value: {current_objective:.6f}")
            print(f"Final proximal residual: {current_prox_residual:.2e}")
            print(f"Final gradient norm: {current_grad_norm:.2e}")
            print(f"Number of iterations: {len(history['iterations'])}")
            print(f"L1 norm of solution: {np.sum(np.abs(x)):.6f}")
            print(
                f"Number of non-zero coefficients: {np.sum(np.abs(x) > 1e-10)}")

        return {
            'solution': x,
            'convergence_history': history,
            'final_objective': current_objective,
            'final_proximal_residual': current_prox_residual,
            'iterations': len(history['iterations'])
        }

    def get_optimality_conditions(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Check optimality conditions for the current iterate.

        Args:
            x: Current iterate

        Returns:
            Dictionary with optimality measures
        """
        # Optimality condition: 0 ∈ ∇f(x*) + ∂g(x*)
        # This means: -∇f(x*) ∈ ∂g(x*)
        # Which is equivalent to: x* = prox_{g}(x* - (1/L)∇f(x*))

        grad_f = self.problem.gradient(x)
        x_temp = x - self.step_size * grad_f
        prox_result = self.problem.l1_proximal_operator(
            x_temp, self.lambda_param * self.step_size)

        proximal_residual = np.linalg.norm(x - prox_result)

        # Check if the subgradient condition is satisfied
        # For L1 norm, the subgradient at x_i is:
        # - sign(x_i) if x_i ≠ 0
        # - [-1, 1] if x_i = 0

        subgradient_violation = 0.0
        for i in range(len(x)):
            if abs(x[i]) > 1e-10:  # Non-zero component
                # Should have: grad_f[i] + λ * sign(x[i]) = 0
                expected_grad = -self.lambda_param * np.sign(x[i])
                violation = abs(grad_f[i] - expected_grad)
                subgradient_violation = max(subgradient_violation, violation)
            else:  # Zero component
                # Should have: |grad_f[i]| ≤ λ
                if abs(grad_f[i]) > self.lambda_param + 1e-10:
                    violation = abs(grad_f[i]) - self.lambda_param
                    subgradient_violation = max(
                        subgradient_violation, violation)

        return {
            'proximal_residual': proximal_residual,
            'subgradient_violation': subgradient_violation,
            'gradient_norm': np.linalg.norm(grad_f),
            'l1_norm': np.sum(np.abs(x))
        }
