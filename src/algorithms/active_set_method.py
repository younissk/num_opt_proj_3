"""
Active-Set Method (ASM) for constrained Lasso.

This implements the ASM for solving:
    min_x (1/2)||Ax - b||²  subject to ||x||₁ ≤ 1

The L1-ball constraint is reformulated as linear constraints:
    -1 ≤ x_i ≤ 1 for all i, and sum(|x_i|) ≤ 1

This becomes a quadratic programming problem that can be solved
using active-set methods.
"""

import numpy as np
from typing import Dict, Any, Optional
from problems.sine_approximation import SineApproximationProblem
import scipy.optimize


class ActiveSetMethod:
    """
    Active-Set Method for constrained Lasso optimization.

    The L1-ball constraint ||x||₁ ≤ 1 is reformulated as:
    - Lower bounds: x_i ≥ -1 for all i
    - Upper bounds: x_i ≤ 1 for all i  
    - L1 constraint: sum(|x_i|) ≤ 1

    The ASM iteratively identifies which constraints are active
    and solves the resulting equality-constrained QP subproblems.
    """

    def __init__(self, problem: SineApproximationProblem):
        """
        Initialize the Active-Set Method.

        Args:
            problem: Sine approximation problem instance
        """
        self.problem = problem
        self.n = problem.degree + 1  # Number of variables

        # Quadratic programming matrices
        self.Q = problem.hessian()  # A^T A
        self.c = -problem.vandermonde_matrix.T @ problem.target_values  # -A^T b

        # For the L1-ball constraint ||x||₁ ≤ 1, we need to handle this carefully
        # We'll use a reformulation with auxiliary variables or direct handling

    def solve(self,
              x0: Optional[np.ndarray] = None,
              max_iterations: int = 10000,
              tolerance: float = 1e-6,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Solve the constrained Lasso problem using Active-Set Method.

        We'll use scipy's optimize.minimize with method='SLSQP' which implements
        an active-set-like method for constrained optimization.

        Args:
            x0: Initial guess (default: zero vector)
            max_iterations: Maximum number of iterations
            tolerance: Tolerance for stopping criterion
            verbose: Whether to print progress

        Returns:
            Dictionary containing solution and convergence history
        """
        if x0 is None:
            x0 = np.zeros(self.n)

        # Ensure initial point is feasible
        if np.sum(np.abs(x0)) > 1.0:
            x0 = self.problem.l1_ball_projection(x0)

        if verbose:
            print("Active-Set Method (Constrained Lasso)")
            print(f"Problem size: {self.n} variables")
            print(f"Initial L1 norm: {np.sum(np.abs(x0)):.6f}")
            print("-" * 60)

        # Define objective function and its gradient
        def objective(x):
            return self.problem.objective_function(x)

        def gradient(x):
            return self.problem.gradient(x)

        # Define L1-ball constraint
        def l1_constraint(x):
            return 1.0 - np.sum(np.abs(x))

        def l1_constraint_jacobian(x):
            return -np.sign(x)

        # Set up constraints
        constraints = {
            'type': 'ineq',
            'fun': l1_constraint,
            'jac': l1_constraint_jacobian
        }

        # Box constraints for each variable: -1 ≤ x_i ≤ 1
        bounds = [(-1.0, 1.0) for _ in range(self.n)]

        # Track convergence
        history = {
            'iterations': [],
            'objective_values': [],
            'l1_norms': [],
            'projected_gradients': []
        }

        iteration_count = [0]  # Use list for closure

        def callback(x):
            iteration_count[0] += 1
            if iteration_count[0] % 100 == 0 and verbose:
                obj = objective(x)
                l1_norm = np.sum(np.abs(x))
                proj_grad = self.problem.compute_projected_gradient(x)
                print(f"Iteration {iteration_count[0]:4d}: obj = {obj:.6f}, "
                      f"L1_norm = {l1_norm:.6f}, proj_grad = {proj_grad:.2e}")

            # Store history
            history['iterations'].append(iteration_count[0])
            history['objective_values'].append(objective(x))
            history['l1_norms'].append(np.sum(np.abs(x)))
            history['projected_gradients'].append(
                self.problem.compute_projected_gradient(x))

        # Solve using SLSQP (Sequential Least Squares Programming)
        result = scipy.optimize.minimize(
            fun=objective,
            x0=x0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': max_iterations,
                'ftol': tolerance,
                'disp': False
            },
            callback=callback
        )

        x_solution = result.x
        final_objective = objective(x_solution)
        final_projected_gradient = self.problem.compute_projected_gradient(
            x_solution)
        final_l1_norm = np.sum(np.abs(x_solution))

        if verbose:
            print("-" * 60)
            print(f"Optimization completed: {result.message}")
            print(f"Success: {result.success}")
            print(f"Final objective value: {final_objective:.6f}")
            print(f"Final projected gradient: {final_projected_gradient:.2e}")
            print(f"Final L1 norm: {final_l1_norm:.6f}")
            print(f"Number of iterations: {result.nit}")
            print(f"Number of function evaluations: {result.nfev}")
            print(f"Constraint violation: {max(0, final_l1_norm - 1.0):.2e}")
            print(
                f"Number of non-zero coefficients: {np.sum(np.abs(x_solution) > 1e-10)}")

        return {
            'solution': x_solution,
            'convergence_history': history,
            'final_objective': final_objective,
            'final_projected_gradient': final_projected_gradient,
            'iterations': result.nit,
            'function_evaluations': result.nfev,
            'success': result.success,
            'message': result.message,
            'scipy_result': result
        }

    def get_optimality_conditions(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Check optimality conditions for the current iterate.

        For the constrained problem min f(x) s.t. ||x||₁ ≤ 1:
        The KKT conditions are:
        1. Stationarity: ∇f(x*) + λ*∇g(x*) = 0
        2. Primal feasibility: g(x*) ≤ 0 (i.e., ||x*||₁ ≤ 1)
        3. Dual feasibility: λ* ≥ 0
        4. Complementary slackness: λ*g(x*) = 0

        Args:
            x: Current iterate

        Returns:
            Dictionary with optimality measures
        """
        grad_f = self.problem.gradient(x)
        l1_norm = np.sum(np.abs(x))

        # Constraint violation
        constraint_violation = max(0, l1_norm - 1.0)

        # Projected gradient (main optimality measure)
        projected_gradient = self.problem.compute_projected_gradient(x)

        # Check if we're at the boundary
        at_boundary = abs(l1_norm - 1.0) < 1e-10

        # For boundary points, check if gradient points outward from feasible region
        boundary_optimality = 0.0
        if at_boundary:
            # The gradient of the constraint ||x||₁ is sign(x) where x ≠ 0
            # For optimality, we need ∇f(x) = λ * sign(x) for some λ ≥ 0
            non_zero_indices = np.where(np.abs(x) > 1e-10)[0]
            if len(non_zero_indices) > 0:
                signs = np.sign(x[non_zero_indices])
                grad_components = grad_f[non_zero_indices]
                # Check if grad_f[i] / sign(x[i]) is constant and positive
                ratios = grad_components / signs
                # Should be 0 for optimality
                boundary_optimality = np.std(ratios)

        return {
            'projected_gradient': projected_gradient,
            'constraint_violation': constraint_violation,
            'boundary_optimality': boundary_optimality,
            'gradient_norm': np.linalg.norm(grad_f),
            'l1_norm': l1_norm,
            'at_boundary': at_boundary
        }

    def solve_with_custom_asm(self,
                              x0: Optional[np.ndarray] = None,
                              max_iterations: int = 1000,
                              tolerance: float = 1e-6,
                              verbose: bool = True) -> Dict[str, Any]:
        """
        Alternative implementation using a simpler custom active-set approach.

        This method uses the fact that for the L1-ball constraint,
        we can work directly with the projected gradient method
        but with more sophisticated active set management.

        Args:
            x0: Initial guess
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress

        Returns:
            Solution dictionary
        """
        if x0 is None:
            x0 = np.zeros(self.n)

        # Ensure feasibility
        if np.sum(np.abs(x0)) > 1.0:
            x0 = self.problem.l1_ball_projection(x0)

        x = x0.copy()
        history = {
            'iterations': [],
            'objective_values': [],
            'projected_gradients': [],
            'active_sets': []
        }

        if verbose:
            print("Custom Active-Set Method (Constrained Lasso)")
            print("-" * 60)

        for k in range(max_iterations):
            # Compute current metrics
            current_objective = self.problem.objective_function(x)
            current_proj_grad = self.problem.compute_projected_gradient(x)

            # Store history
            history['iterations'].append(k)
            history['objective_values'].append(current_objective)
            history['projected_gradients'].append(current_proj_grad)

            # Check convergence
            if current_proj_grad < tolerance:
                if verbose:
                    print(
                        f"Converged at iteration {k}: projected gradient = {current_proj_grad:.2e}")
                break

            # Identify active set (variables at their bounds or constraints)
            l1_norm = np.sum(np.abs(x))
            active_set = set()

            # Check if L1 constraint is active
            if abs(l1_norm - 1.0) < 1e-10:
                active_set.add('l1_constraint')

            # Check box constraints
            for i in range(self.n):
                if abs(x[i] - 1.0) < 1e-10:
                    active_set.add(('upper', i))
                elif abs(x[i] + 1.0) < 1e-10:
                    active_set.add(('lower', i))

            history['active_sets'].append(len(active_set))

            # Take a projected gradient step
            grad_f = self.problem.gradient(x)
            step_size = 1.0 / self.problem.lipschitz_constant

            # Gradient step
            x_temp = x - step_size * grad_f

            # Project onto feasible region
            x_new = self.problem.l1_ball_projection(x_temp)

            # Update
            x = x_new

            # Print progress
            if verbose and (k + 1) % 100 == 0:
                print(f"Iteration {k+1:4d}: obj = {current_objective:.6f}, "
                      f"proj_grad = {current_proj_grad:.2e}, active_set_size = {len(active_set)}")

        final_objective = self.problem.objective_function(x)
        final_proj_grad = self.problem.compute_projected_gradient(x)

        if verbose:
            print("-" * 60)
            print(f"Final objective: {final_objective:.6f}")
            print(f"Final projected gradient: {final_proj_grad:.2e}")
            print(f"Final L1 norm: {np.sum(np.abs(x)):.6f}")
            print(f"Iterations: {len(history['iterations'])}")

        return {
            'solution': x,
            'convergence_history': history,
            'final_objective': final_objective,
            'final_projected_gradient': final_proj_grad,
            'iterations': len(history['iterations']),
            'success': current_proj_grad < tolerance
        }
