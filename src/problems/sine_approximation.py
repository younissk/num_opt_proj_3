"""
Sine function approximation problem using polynomial regression.

This module implements the core problem formulation for approximating sin(t) 
over [-2π, 2π] using a polynomial of degree n with Lasso regularization.
"""

import numpy as np
import matplotlib.pyplot as plt
from .base_problem import BaseOptimizationProblem


class SineApproximationProblem(BaseOptimizationProblem):
    """
    Problem formulation for approximating sin(t) using polynomial regression.

    The problem minimizes f(x) = 1/2 * ||Ax - b||² where:
    - A is the Vandermonde matrix for polynomial basis
    - b contains sin(a_j) values at sample points a_j
    - x is the coefficient vector for the polynomial
    """

    def __init__(self, degree: int = 10, num_samples: int = 100):
        """
        Initialize the sine approximation problem.

        Args:
            degree: Degree of the polynomial (n)
            num_samples: Number of sample points (m)
        """
        self._degree = degree
        self._num_samples = num_samples

        # Generate sample points uniformly spaced over [-2π, 2π]
        self.sample_points = np.linspace(-2*np.pi, 2*np.pi, num_samples)

        # Target values: b_j = sin(a_j)
        self._target_values = np.sin(self.sample_points)

        # Construct Vandermonde matrix A
        self._vandermonde_matrix = self._construct_vandermonde_matrix()

        # Compute Lipschitz constant L = ||A^T A||_2
        self._lipschitz_constant = self._compute_lipschitz_constant()

        # Condition number of A^T A
        self._condition_number = self._compute_condition_number()

    @property
    def degree(self) -> int:
        """Return the polynomial degree."""
        return self._degree
    
    @property
    def num_samples(self) -> int:
        """Return the number of sample points."""
        return self._num_samples
    
    @property
    def lipschitz_constant(self) -> float:
        """Return the Lipschitz constant of the gradient."""
        return self._lipschitz_constant
    
    @property
    def condition_number(self) -> float:
        """Return the condition number of A^T A."""
        return self._condition_number
    
    @property
    def vandermonde_matrix(self) -> np.ndarray:
        """Return the Vandermonde matrix."""
        return self._vandermonde_matrix
    
    @property
    def target_values(self) -> np.ndarray:
        """Return the target values."""
        return self._target_values

    def _construct_vandermonde_matrix(self) -> np.ndarray:
        """
        Construct the Vandermonde matrix A where A[j,i] = a_j^i.

        Returns:
            Vandermonde matrix of shape (num_samples, degree + 1)
        """
        # Each row j corresponds to sample point a_j
        # Each column i corresponds to power i (0 to degree)
        matrix = np.zeros((self.num_samples, self.degree + 1))

        for j, a_j in enumerate(self.sample_points):
            for i in range(self.degree + 1):
                matrix[j, i] = a_j ** i

        return matrix

    def _compute_lipschitz_constant(self) -> float:
        """
        Compute the Lipschitz constant L = ||A^T A||_2.

        Returns:
            Largest singular value of A^T A
        """
        ATA = self._vandermonde_matrix.T @ self._vandermonde_matrix
        singular_values = np.linalg.svd(ATA, compute_uv=False)
        return singular_values[0]  # Largest singular value

    def _compute_condition_number(self) -> float:
        """
        Compute the condition number of A^T A.

        Returns:
            Ratio of largest to smallest singular value
        """
        ATA = self._vandermonde_matrix.T @ self._vandermonde_matrix
        singular_values = np.linalg.svd(ATA, compute_uv=False)
        return singular_values[0] / singular_values[-1]

    def objective_function(self, x: np.ndarray) -> float:
        """
        Compute the objective function value f(x) = 1/2 * ||Ax - b||².

        Args:
            x: Coefficient vector of shape (degree + 1,)

        Returns:
            Objective function value
        """
        residual = self._vandermonde_matrix @ x - self._target_values
        return 0.5 * np.sum(residual ** 2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient ∇f(x) = A^T (Ax - b).

        Args:
            x: Coefficient vector of shape (degree + 1,)

        Returns:
            Gradient vector of shape (degree + 1,)
        """
        residual = self._vandermonde_matrix @ x - self._target_values
        return self._vandermonde_matrix.T @ residual

    def hessian(self) -> np.ndarray:
        """
        Compute the Hessian ∇²f(x) = A^T A.

        Returns:
            Hessian matrix of shape (degree + 1, degree + 1)
        """
        return self._vandermonde_matrix.T @ self._vandermonde_matrix

    def evaluate_polynomial(self, x: np.ndarray, t_values: np.ndarray) -> np.ndarray:
        """
        Evaluate the polynomial φ(x; t) = Σᵢ₌₀ⁿ xᵢ tⁱ at given t values.

        Args:
            x: Coefficient vector of shape (degree + 1,)
            t_values: Points at which to evaluate the polynomial

        Returns:
            Polynomial values at t_values
        """
        result = np.zeros_like(t_values)
        for i, coeff in enumerate(x):
            result += coeff * (t_values ** i)
        return result

    def plot_approximation(self, x: np.ndarray, title: str = "Sine Function Approximation") -> None:
        """
        Plot the original sine function and its polynomial approximation.

        Args:
            x: Coefficient vector for the polynomial
            title: Plot title
        """
        # Fine grid for smooth plotting
        t_fine = np.linspace(-2*np.pi, 2*np.pi, 1000)

        # Original sine function
        y_true = np.sin(t_fine)

        # Polynomial approximation
        y_approx = self.evaluate_polynomial(x, t_fine)

        # Sample points used for fitting
        y_samples = self.target_values

        plt.figure(figsize=(10, 6))
        plt.plot(t_fine, y_true, 'b-', label='True sin(t)', linewidth=2)
        plt.plot(t_fine, y_approx, 'r--',
                 label=f'Polynomial (deg={self.degree})', linewidth=2)
        plt.plot(self.sample_points, y_samples, 'ko',
                 label='Sample points', markersize=4)

        plt.xlabel('t')
        plt.ylabel('y')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_problem_info(self) -> dict:
        """
        Get summary information about the problem.

        Returns:
            Dictionary with problem parameters and properties
        """
        return {
            'degree': self.degree,
            'num_samples': self.num_samples,
            'matrix_shape': self.vandermonde_matrix.shape,
            'lipschitz_constant': self.lipschitz_constant,
            'condition_number': self.condition_number,
            'sample_range': [-2*np.pi, 2*np.pi]
        }

    def l1_proximal_operator(self, x: np.ndarray, lambda_param: float) -> np.ndarray:
        """
        Compute the L1-norm proximal operator (soft-thresholding).

        For the function g(x) = λ||x||₁, the proximal operator is:
        prox_{λ||·||₁}(x) = sign(x) ⊙ max(|x| - λ, 0)

        Args:
            x: Input vector
            lambda_param: Regularization parameter λ > 0

        Returns:
            Proximal operator result
        """
        return np.sign(x) * np.maximum(np.abs(x) - lambda_param, 0)

    def l1_ball_projection(self, x: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """
        Project vector x onto the L1-ball {z : ||z||₁ ≤ radius}.

        This is the solution to: min_{||z||₁ ≤ radius} ||z - x||²

        Args:
            x: Input vector to project
            radius: L1-ball radius (default: 1.0)

        Returns:
            Projected vector
        """
        if np.sum(np.abs(x)) <= radius:
            return x.copy()

        # Sort absolute values in descending order
        abs_x = np.abs(x)
        sorted_indices = np.argsort(abs_x)[::-1]
        sorted_abs = abs_x[sorted_indices]

        # Find the largest k such that sum of top k elements > radius
        cumulative_sum = np.cumsum(sorted_abs)
        k = np.searchsorted(cumulative_sum, radius)

        if k == 0:
            # All elements are zero
            return np.zeros_like(x)

        # Compute threshold
        if k < len(x):
            threshold = sorted_abs[k-1] - (cumulative_sum[k-1] - radius) / k
        else:
            threshold = 0

        # Apply soft-thresholding
        result = np.zeros_like(x)
        for i, idx in enumerate(sorted_indices):
            if i < k:
                result[idx] = np.sign(x[idx]) * \
                    max(float(abs_x[idx] - threshold), 0.0)

        return result

    def compute_proximal_residual(self, x: np.ndarray, lambda_param: float) -> float:
        """
        Compute the proximal residual for the penalized Lasso problem.

        The proximal residual measures how close x is to being optimal:
        ||x - prox_{λ||·||₁}(x - (1/L)∇f(x))||

        Args:
            x: Current iterate
            lambda_param: Regularization parameter

        Returns:
            Proximal residual norm
        """
        step_size = 1.0 / self.lipschitz_constant
        grad_f = self.gradient(x)
        x_temp = x - step_size * grad_f
        prox_result = self.l1_proximal_operator(
            x_temp, lambda_param * step_size)
        return float(np.linalg.norm(x - prox_result))

    def compute_projected_gradient(self, x: np.ndarray) -> float:
        """
        Compute the projected gradient norm for the constrained Lasso problem.

        The projected gradient measures optimality for the constrained problem:
        ||x - P_{||·||₁ ≤ 1}(x - (1/L)∇f(x))||

        Args:
            x: Current iterate

        Returns:
            Projected gradient norm
        """
        step_size = 1.0 / self.lipschitz_constant
        grad_f = self.gradient(x)
        x_temp = x - step_size * grad_f
        proj_result = self.l1_ball_projection(x_temp)
        return float(np.linalg.norm(x - proj_result))
