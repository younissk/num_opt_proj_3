"""
Projected Gradient algorithm for constrained Lasso.

This implements the PG method for solving:
    min_x (1/2)||Ax - b||²  subject to ||x||₁ ≤ 1

The algorithm alternates between:
1. Gradient step: x_temp = x^k - (1/L)∇f(x^k)
2. Projection step: x^{k+1} = P_{||·||₁ ≤ 1}(x_temp)
"""

import numpy as np
from typing import Dict, Any, Optional
from problems.base_problem import BaseOptimizationProblem


class ProjectedGradient:
    """
    Projected Gradient algorithm for constrained Lasso optimization.
    
    Algorithm:
        x^{k+1} = P_{||·||₁ ≤ 1}(x^k - (1/L)∇f(x^k))
    
    where:
        - f(x) = (1/2)||Ax - b||² (objective function)
        - P_{||·||₁ ≤ 1} is the projection onto the L1-ball
        - L is the Lipschitz constant of ∇f
    """
    
    def __init__(self, problem: BaseOptimizationProblem):
        """
        Initialize the Projected Gradient algorithm.
        
        Args:
            problem: Sine approximation problem instance
        """
        self.problem = problem
        self.step_size = 1.0 / problem.lipschitz_constant
        
        # Optimality conditions:
        # For the constrained Lasso: min f(x) subject to ||x||₁ ≤ 1
        # The optimality condition is: ∇f(x*) ∈ N_C(x*)
        # where N_C(x*) is the normal cone to the constraint set at x*
        # This means: x* = P_C(x* - (1/L)∇f(x*))
        # where P_C is the projection onto the constraint set
        # The projected gradient residual measures this condition
        # ||x - P_C(x - (1/L)∇f(x))|| = 0 at optimality
    
    def solve(self, 
              x0: Optional[np.ndarray] = None, 
              max_iterations: int = 100000,
              tolerance: float = 1e-6,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Solve the constrained Lasso problem using Projected Gradient.
        
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
        
        # Ensure initial point is feasible
        if np.sum(np.abs(x0)) > 1.0:
            x0 = self.problem.l1_ball_projection(x0)
        
        x = x0.copy()
        history = {
            'iterations': [],
            'objective_values': [],
            'projected_gradients': [],
            'gradient_norms': [],
            'l1_norms': []
        }
        
        if verbose:
            print("Projected Gradient Algorithm (Constrained Lasso)")
            print(f"Step size: {self.step_size:.2e}")
            print(f"Lipschitz constant: {self.problem.lipschitz_constant:.2e}")
            print("-" * 60)
        
        for k in range(max_iterations):
            # Store current state
            current_objective = self.problem.objective_function(x)
            current_proj_grad = self.problem.compute_projected_gradient(x)
            current_grad_norm = np.linalg.norm(self.problem.gradient(x))
            current_l1_norm = np.sum(np.abs(x))
            
            history['iterations'].append(k)
            history['objective_values'].append(current_objective)
            history['projected_gradients'].append(current_proj_grad)
            history['gradient_norms'].append(current_grad_norm)
            history['l1_norms'].append(current_l1_norm)
            
            # Check stopping criterion
            if current_proj_grad < tolerance:
                if verbose:
                    print(f"Converged at iteration {k}: projected gradient = {current_proj_grad:.2e}")
                break
            
            # Gradient step: x_temp = x - (1/L)∇f(x)
            grad_f = self.problem.gradient(x)
            x_temp = x - self.step_size * grad_f
            
            # Projection step: x_new = P_{||·||₁ ≤ 1}(x_temp)
            x_new = self.problem.l1_ball_projection(x_temp)
            
            # Update iterate
            x = x_new
            
            # Print progress every 1000 iterations
            if verbose and (k + 1) % 1000 == 0:
                print(f"Iteration {k+1:6d}: obj = {current_objective:.6f}, "
                      f"proj_grad = {current_proj_grad:.2e}, L1_norm = {current_l1_norm:.6f}")
        
        if verbose:
            print("-" * 60)
            print(f"Final objective value: {current_objective:.6f}")
            print(f"Final projected gradient: {current_proj_grad:.2e}")
            print(f"Final gradient norm: {current_grad_norm:.2e}")
            print(f"Final L1 norm: {current_l1_norm:.6f}")
            print(f"Number of iterations: {len(history['iterations'])}")
            print(f"Number of non-zero coefficients: {np.sum(np.abs(x) > 1e-10)}")
        
        return {
            'solution': x,
            'convergence_history': history,
            'final_objective': current_objective,
            'final_projected_gradient': current_proj_grad,
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
        # Optimality condition for constrained optimization:
        # ∇f(x*) ∈ N_C(x*) where N_C(x*) is the normal cone
        
        # For the L1-ball constraint ||x||₁ ≤ 1:
        # - If ||x||₁ < 1 (interior point): N_C(x) = {0}, so ∇f(x*) = 0
        # - If ||x||₁ = 1 (boundary point): N_C(x) = {λ * sign(x_i) : λ ≥ 0}
        
        grad_f = self.problem.gradient(x)
        x_temp = x - self.step_size * grad_f
        proj_result = self.problem.l1_ball_projection(x_temp)
        
        projected_gradient = np.linalg.norm(x - proj_result)
        
        # Check constraint satisfaction
        l1_norm = np.sum(np.abs(x))
        constraint_violation = max(0, l1_norm - 1.0)
        
        # Check optimality at boundary points
        boundary_optimality = 0.0
        if abs(l1_norm - 1.0) < 1e-10:  # On the boundary
            # Should have: ∇f(x*) = λ * sign(x*) for some λ ≥ 0
            # This means: sign(x_i) * grad_f[i] should be constant for non-zero x_i
            non_zero_indices = np.where(np.abs(x) > 1e-10)[0]
            if len(non_zero_indices) > 0:
                signs = np.sign(x[non_zero_indices])
                grad_signs = signs * grad_f[non_zero_indices]
                # All grad_signs should be equal (up to numerical precision)
                max_grad_sign = np.max(grad_signs)
                min_grad_sign = np.min(grad_signs)
                boundary_optimality = max_grad_sign - min_grad_sign
        
        return {
            'projected_gradient': projected_gradient,
            'constraint_violation': constraint_violation,
            'boundary_optimality': boundary_optimality,
            'gradient_norm': np.linalg.norm(grad_f),
            'l1_norm': l1_norm
        }
