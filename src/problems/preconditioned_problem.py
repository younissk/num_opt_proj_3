"""
Pre-conditioned sine approximation problem.

This module implements various pre-conditioning strategies to address
the numerical conditioning issues in the original Vandermonde matrix.
"""

import numpy as np
from typing import Dict, Any
from .sine_approximation import SineApproximationProblem
from .base_problem import BaseOptimizationProblem


class PreconditionedSineProblem(BaseOptimizationProblem):
    """
    Pre-conditioned version of the sine approximation problem.

    The original problem has condition number κ(A^T A) that grows exponentially
    with polynomial degree. This class implements various pre-conditioning
    strategies to improve numerical stability.
    """

    def __init__(self, original_problem: SineApproximationProblem,
                 preconditioning_method: str = 'cholesky'):
        """
        Initialize the pre-conditioned problem.

        Args:
            original_problem: The original sine approximation problem
            preconditioning_method: Method to use ('cholesky', 'svd', 'diagonal')
        """
        self.original_problem = original_problem
        self.preconditioning_method = preconditioning_method

        # Original matrices
        self.A_original = original_problem.vandermonde_matrix
        self.b_original = original_problem.target_values
        self.n = original_problem.degree + 1
        self.m = original_problem.num_samples

        # Apply pre-conditioning
        self._apply_preconditioning()

        # Compute new problem properties
        self._compute_problem_properties()

    def _apply_preconditioning(self):
        """Apply the selected pre-conditioning method."""
        if self.preconditioning_method == 'cholesky':
            self._cholesky_preconditioning()
        elif self.preconditioning_method == 'svd':
            self._svd_preconditioning()
        elif self.preconditioning_method == 'diagonal':
            self._diagonal_preconditioning()
        else:
            raise ValueError(
                f"Unknown pre-conditioning method: {self.preconditioning_method}")

    def _cholesky_preconditioning(self):
        """
        Cholesky-based pre-conditioning.

        This method computes the Cholesky decomposition of A^T A and uses it
        to transform the problem. The transformed problem has better conditioning.
        """
        # Compute A^T A
        ATA = self.A_original.T @ self.A_original

        try:
            # Attempt Cholesky decomposition
            L = np.linalg.cholesky(ATA)
            L_inv = np.linalg.inv(L)

            # Transform the problem: A_new = A * L^(-1), x_new = L * x
            self.A_preconditioned = self.A_original @ L_inv
            self.b_preconditioned = self.b_original.copy()
            self.L_matrix = L
            self.L_inv_matrix = L_inv

            # Transformation matrices
            self.P = L_inv  # Pre-conditioner
            self.P_inv = L  # Inverse pre-conditioner

            self.preconditioning_info = {
                'method': 'cholesky',
                'success': True,
                'L_norm': np.linalg.norm(L),
                'L_inv_norm': np.linalg.norm(L_inv)
            }

        except np.linalg.LinAlgError:
            # If Cholesky fails (not positive definite), fall back to diagonal
            print(
                "Cholesky decomposition failed, falling back to diagonal pre-conditioning")
            self._diagonal_preconditioning()

    def _svd_preconditioning(self):
        """
        SVD-based pre-conditioning.

        This method uses the singular value decomposition to create a
        well-conditioned transformation of the problem.
        """
        # Compute SVD of A
        U, s, Vt = np.linalg.svd(self.A_original, full_matrices=False)

        # Create diagonal scaling matrix
        # Scale by inverse of singular values, but with regularization
        epsilon = 1e-12
        s_scaled = np.sqrt(1.0 / (s**2 + epsilon))

        # Create pre-conditioner: P = V * diag(s_scaled)
        P = Vt.T * s_scaled.reshape(1, -1)
        P_inv = Vt.T * (1.0 / s_scaled).reshape(1, -1)

        # Transform the problem
        self.A_preconditioned = self.A_original @ P
        self.b_preconditioned = self.b_original.copy()
        self.P = P
        self.P_inv = P_inv

        # Store SVD components
        self.U = U
        self.s = s
        self.Vt = Vt
        self.s_scaled = s_scaled

        self.preconditioning_info = {
            'method': 'svd',
            'success': True,
            'singular_values': s,
            'scaled_singular_values': s_scaled,
            'min_singular_value': np.min(s),
            'max_singular_value': np.max(s)
        }

    def _diagonal_preconditioning(self):
        """
        Diagonal pre-conditioning.

        This method scales the columns of A to have unit norm, which
        is a simple but effective pre-conditioning strategy.
        """
        # Compute column norms
        col_norms = np.linalg.norm(self.A_original, axis=0)

        # Avoid division by zero
        col_norms = np.maximum(col_norms, 1e-12)

        # Create diagonal pre-conditioner
        P = np.diag(1.0 / col_norms)
        P_inv = np.diag(col_norms)

        # Transform the problem
        self.A_preconditioned = self.A_original @ P
        self.b_preconditioned = self.b_original.copy()
        self.P = P
        self.P_inv = P_inv

        self.preconditioning_info = {
            'method': 'diagonal',
            'success': True,
            'column_norms': col_norms,
            'min_norm': np.min(col_norms),
            'max_norm': np.max(col_norms)
        }

    def _compute_problem_properties(self):
        """Compute properties of the pre-conditioned problem."""
        # Compute new condition number
        ATA_new = self.A_preconditioned.T @ self.A_preconditioned
        singular_values_new = np.linalg.svd(ATA_new, compute_uv=False)
        self._condition_number = singular_values_new[0] / \
            singular_values_new[-1]

        # Compute new Lipschitz constant
        self._lipschitz_constant = singular_values_new[0]

        # Compute improvement metrics
        original_condition = self.original_problem.condition_number
        self.condition_improvement = original_condition / self.condition_number

        # Store transformation matrices
        self.ATA_original = self.A_original.T @ self.A_original
        self.ATA_preconditioned = ATA_new

    @property
    def degree(self) -> int:
        """Return the polynomial degree."""
        return self.original_problem.degree

    @property
    def num_samples(self) -> int:
        """Return the number of sample points."""
        return self.original_problem.num_samples

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
        """Return the Vandermonde matrix (pre-conditioned version)."""
        return self.A_preconditioned

    @property
    def target_values(self) -> np.ndarray:
        """Return the target values (pre-conditioned version)."""
        return self.b_preconditioned

    def objective_function(self, x: np.ndarray) -> float:
        """
        Compute the objective function value for the pre-conditioned problem.

        Args:
            x: Coefficient vector in the pre-conditioned space

        Returns:
            Objective function value
        """
        residual = self.A_preconditioned @ x - self.b_preconditioned
        return float(0.5 * np.sum(residual ** 2))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient for the pre-conditioned problem.

        Args:
            x: Coefficient vector in the pre-conditioned space

        Returns:
            Gradient vector
        """
        residual = self.A_preconditioned @ x - self.b_preconditioned
        return self.A_preconditioned.T @ residual

    def hessian(self) -> np.ndarray:
        """
        Compute the Hessian for the pre-conditioned problem.

        Returns:
            Hessian matrix
        """
        return self.A_preconditioned.T @ self.A_preconditioned

    def transform_solution(self, x_preconditioned: np.ndarray) -> np.ndarray:
        """
        Transform solution from pre-conditioned space back to original space.

        Args:
            x_preconditioned: Solution in pre-conditioned space

        Returns:
            Solution in original space
        """
        return self.P @ x_preconditioned

    def transform_gradient(self, grad_preconditioned: np.ndarray) -> np.ndarray:
        """
        Transform gradient from pre-conditioned space back to original space.

        Args:
            grad_preconditioned: Gradient in pre-conditioned space

        Returns:
            Gradient in original space
        """
        return self.P @ grad_preconditioned

    def evaluate_polynomial(self, x_preconditioned: np.ndarray, t_values: np.ndarray) -> np.ndarray:
        """
        Evaluate the polynomial using pre-conditioned coefficients.

        Args:
            x_preconditioned: Coefficient vector in pre-conditioned space
            t_values: Points at which to evaluate the polynomial

        Returns:
            Polynomial values at t_values
        """
        # Transform to original space first
        x_original = self.transform_solution(x_preconditioned)

        # Use original problem's evaluation method
        return self.original_problem.evaluate_polynomial(x_original, t_values)

    def get_preconditioning_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pre-conditioning results.

        Returns:
            Dictionary with pre-conditioning information
        """
        original_condition = self.original_problem.condition_number

        return {
            'original_condition_number': original_condition,
            'preconditioned_condition_number': self.condition_number,
            'condition_improvement': self.condition_improvement,
            'condition_improvement_log10': np.log10(self.condition_improvement),
            'method': self.preconditioning_method,
            'preconditioning_info': self.preconditioning_info,
            'lipschitz_constant': self.lipschitz_constant,
            'matrix_dimensions': {
                'A_original': self.A_original.shape,
                'A_preconditioned': self.A_preconditioned.shape
            }
        }

    def plot_conditioning_comparison(self) -> None:
        """Plot the conditioning comparison between original and pre-conditioned problems."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Singular value comparison
        ax = axes[0, 0]
        s_orig = np.linalg.svd(self.ATA_original, compute_uv=False)
        s_prec = np.linalg.svd(self.ATA_preconditioned, compute_uv=False)

        ax.semilogy(range(1, len(s_orig) + 1), s_orig, 'bo-',
                    label='Original A^T A', linewidth=2, markersize=6)
        ax.semilogy(range(1, len(s_prec) + 1), s_prec, 'ro-',
                    label='Pre-conditioned A^T A', linewidth=2, markersize=6)
        ax.set_xlabel('Index')
        ax.set_ylabel('Singular Value')
        ax.set_title('Singular Value Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Condition number improvement
        ax = axes[0, 1]
        methods = ['Original', 'Pre-conditioned']
        condition_numbers = [
            self.original_problem.condition_number, self.condition_number]

        bars = ax.bar(methods, condition_numbers,
                      color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel('Condition Number')
        ax.set_title('Condition Number Comparison')
        ax.set_yscale('log')

        # Add value labels on bars
        for bar, value in zip(bars, condition_numbers):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2e}', ha='center', va='bottom')

        # Plot 3: Matrix visualization (original)
        ax = axes[1, 0]
        im1 = ax.imshow(np.log10(np.abs(self.ATA_original)), cmap='viridis')
        ax.set_title('Original A^T A (log scale)')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        plt.colorbar(im1, ax=ax)

        # Plot 4: Matrix visualization (pre-conditioned)
        ax = axes[1, 1]
        im2 = ax.imshow(
            np.log10(np.abs(self.ATA_preconditioned)), cmap='viridis')
        ax.set_title('Pre-conditioned A^T A (log scale)')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        plt.colorbar(im2, ax=ax)

        plt.tight_layout()
        plt.savefig(f'results/plots/preconditioning_{self.preconditioning_method}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def l1_proximal_operator(self, x: np.ndarray, lambda_param: float) -> np.ndarray:
        """
        Compute the L1 proximal operator for the pre-conditioned problem.

        Args:
            x: Coefficient vector in pre-conditioned space
            lambda_param: Regularization parameter

        Returns:
            Result of proximal operator
        """
        # Transform to original space, apply proximal operator, then transform back
        x_original = self.transform_solution(x)
        prox_original = self.original_problem.l1_proximal_operator(
            x_original, lambda_param)
        return self.transform_solution(prox_original)

    def l1_ball_projection(self, x: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """
        Project x onto the L1-ball in pre-conditioned space.

        Args:
            x: Coefficient vector in pre-conditioned space
            radius: Radius of the L1-ball

        Returns:
            Projected vector
        """
        # Transform to original space, project, then transform back
        x_original = self.transform_solution(x)
        proj_original = self.original_problem.l1_ball_projection(
            x_original, radius)
        return self.transform_solution(proj_original)

    def compute_proximal_residual(self, x: np.ndarray, lambda_param: float) -> float:
        """
        Compute the proximal residual for optimality checking.

        Args:
            x: Coefficient vector in pre-conditioned space
            lambda_param: Regularization parameter

        Returns:
            Proximal residual value
        """
        # Transform to original space, compute residual, then transform back
        x_original = self.transform_solution(x)
        residual_original = self.original_problem.compute_proximal_residual(
            x_original, lambda_param)
        return residual_original

    def compute_projected_gradient(self, x: np.ndarray) -> float:
        """
        Compute the projected gradient for optimality checking.

        Args:
            x: Coefficient vector in pre-conditioned space

        Returns:
            Projected gradient value
        """
        # Transform to original space, compute projected gradient, then transform back
        x_original = self.transform_solution(x)
        proj_grad_original = self.original_problem.compute_projected_gradient(
            x_original)
        return proj_grad_original

    def get_problem_info(self) -> Dict[str, Any]:
        """Get comprehensive problem information."""
        return {
            'degree': self.original_problem.degree,
            'num_samples': self.original_problem.num_samples,
            'preconditioning_method': self.preconditioning_method,
            'condition_number': self.condition_number,
            'lipschitz_constant': self.lipschitz_constant,
            'preconditioning_summary': self.get_preconditioning_summary()
        }
